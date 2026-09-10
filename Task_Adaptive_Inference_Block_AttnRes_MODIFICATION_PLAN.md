# Task-Adaptive Inference Block AttnRes — 现有项目修改计划

> **用途**
>
> 本文件用于指导 Codex **修改现有项目**，不是从零实现新项目。
>
> 所有修改必须以当前仓库已有代码为基础：
>
> - 保留已经正确、可复用、通过验证的实现；
> - 只修改与当前正式方法论冲突的部分；
> - 不进行无关重构；
> - 不新建平行替代实现来绕开旧代码；
> - 不为了让 pipeline 跑通而修改研究语义。
>
> 方法论上位约束：
>
> `Task_Adaptive_Inference_Block_AttnRes_SPEC_CN.md`
>
> 若本文件与 SPEC 冲突，以 SPEC 为准。

---

# 1. 修改目标

将当前已有项目从旧版后训练逻辑修改为：

```text
Original pretrained model
↓
ordinary-residual task-specific Discovery
↓
freeze P_task
↓
insert Kimi-compatible Block AttnRes
↓
task-specific Q_t + zero-init Alpha_t
↓
full-parameter joint post-training
↓
Probe
↓
routed inference / evaluation
```

本次修改不是重新设计整个工程。

核心只处理以下旧问题：

1. Discovery 中错误依赖 AttnRes / query / compressed replay；
2. 正式后训练冻结 backbone、只训练 pseudo-query；
3. 缺少 zero-init alpha 的 identity-preserving conversion；
4. 主方法 block 内语义没有被强约束为 Kimi cumulative residual flow；
5. task-specific partition / query / alpha 的加载和隔离需要统一；
6. Probe / inference 必须继续保持从原始输入、layer 0 重新正式前向。

---

# 2. 总体修改原则

Codex 开始修改前，必须先审计现有仓库。

不要假设文件名、类名、目录结构或当前实现状态。

先定位实际存在的：

```text
Discovery entry
Discovery cost implementation
DP search
partition serialization
Block AttnRes implementation
Kimi / Fixed baseline implementation
pseudo-query implementation
training entry
optimizer construction
Probe implementation
evaluation entry
checkpoint format
```

然后基于真实代码制定最小修改集。

禁止：

```text
因为旧代码复杂而直接重写整个项目
复制一套新 pipeline 与旧 pipeline 并存
大规模移动目录
无关命名重构
重写已经验证正确的 dataset / evaluation 代码
```

---

# 3. 第一步：现有代码审计

先只读，不修改。

输出一份：

```text
CURRENT_IMPLEMENTATION_AUDIT.md
```

至少记录：

```text
1. 当前 Discovery 从哪个入口启动
2. Discovery 是否调用 AttnRes
3. Discovery 是否读取 query
4. Discovery 是否执行 true/compressed Block AttnRes replay
5. 当前 DP / partition search 在哪里
6. 当前 Kimi-compatible block accumulation 在哪里
7. 当前主方法是否与 Kimi block flow 共用实现
8. 当前 query 参数如何定义
9. 当前 backbone 是否被冻结
10. 当前 optimizer 有哪些 parameter groups
11. 当前是否已有 alpha / gate
12. Probe 在哪里
13. inference 是否丢弃 Probe state 并从 layer 0 重跑
14. 当前 checkpoint 保存哪些 task-specific state
15. Fixed/Kimi baseline 与主方法是否隔离
```

必须区分：

```text
正确实现
需要修改
需要删除
暂时无法确认
```

不要在审计阶段顺手修改代码。

---

# 4. Discovery 修改

这是优先级最高的修改。

## 4.1 必须保留

保留现有已经正确的：

```text
task-specific dataset split
stable IDs
cost-matrix plumbing
DP search
partition structural constraints
deterministic tie-break
partition serialization
partition hash
task isolation
```

只要这些部分不依赖错误 AttnRes 语义，就不要重写。

---

## 4.2 必须删除或禁用

定位并移除 Discovery 中所有：

```text
Full AttnRes forward
Block AttnRes forward
main-method forward
Kimi Fixed Block forward
pseudo-query access
Q_full access
Q_task access
random query
zero query
alpha
AttnRes softmax
completed AttnRes block state
partial AttnRes block state
true MoiraiBlock compressed replay
```

尤其检查旧逻辑是否存在：

```text
candidate P
↓
load P into main method
↓
run compressed Block AttnRes
↓
score candidate
```

如果存在，必须删除或隔离为 legacy code，正式 pipeline 不得再调用。

---

## 4.3 Discovery reference 必须改为原始模型

Discovery reference model 必须是：

```text
original pretrained model
```

运行：

```text
ordinary residual forward
```

不得在加载 checkpoint 后自动转换成 AttnRes。

增加硬断言：

```text
Discovery mode:
    attnres_enabled == false
    query parameters not created/accessed
    alpha parameters not created/accessed
```

---

## 4.4 Discovery cost 处理

先检查当前仓库是否已有真正的：

```text
ordinary-residual-only candidate cost
ordinary-residual-only global scoring
ordinary-residual-only boundary refinement score
```

如果已有：

```text
复用
验证
补测试
```

不要重写。

如果没有：

```text
STOP
status = RESIDUAL_DISCOVERY_COST_UNDEFINED
```

只报告缺失位置和旧实现依赖链。

Codex 不得自行发明：

```text
新的 residual compression operator
zero-query replay
Fixed Block replay
Full AttnRes replay
```

来代替。

---

# 5. Partition search 修改

如果现有 DP 与当前 SPEC 一致，原则上不改算法。

只检查：

```text
partition unit = complete Transformer Block
continuous
non-overlapping
full coverage
configured block-length constraints
configured singleton constraints
configured candidate N range
deterministic tie-break
```

所有模型规模相关范围必须来自 config / model metadata。

禁止保留类似：

```text
num_layers = 28
num_layers = 32
N = ...
block_size = ...
```

作为主方法硬编码。

如果某些数字只属于某个实验配置，则移动到对应 config，不改成全局常量。

---

# 6. 明确 Kimi Block AttnRes 语义

检查当前主方法是否真正复用或严格等价于 Kimi-compatible block residual flow。

正式主方法必须保持：

```text
embedding independent source
completed block representations
current partial block representation
sequential cumulative residual accumulation inside block
commit partial block at boundary
same source/routing semantics as Kimi-compatible Block AttnRes
```

主方法唯一核心结构变化是：

```text
fixed block boundary
→
task-specific discovered boundary
```

不得出现：

```text
mean pooling
learnable block summary
projection summary
attention pooling inside block
weighted block averaging
new residual source definition
extra block encoder
```

如果主方法和 Kimi baseline 当前存在两套不同 block-accumulation 实现：

优先重构为共享同一个 verified low-level block accumulation primitive，

但不要把 baseline 和主方法的：

```text
config
partition source
checkpoint
output
training entry
```

混在一起。

---

# 7. Partition 生命周期修改

正式 Discovery 完成后保存：

```text
P_task
partition_hash
```

然后冻结。

删除或禁用任何：

```text
post-training rediscovery
P0 -> train -> P1
boundary update during training
partition tournament
alternating partition/training
```

训练前后必须验证：

```text
partition_hash_before == partition_hash_after
```

---

# 8. Block AttnRes conversion 修改

只有在加载最终 `P_task` 后，才允许创建正式 Block AttnRes runtime。

修改现有 conversion / model wrapper，使其：

```text
load original pretrained backbone
load frozen P_task
instantiate Kimi-compatible Block AttnRes
create task-specific Q_t
create task-specific Alpha_t
```

不得让 Discovery 复用这个 runtime conversion。

---

# 9. Alpha 修改

如果项目当前没有 alpha，增加：

```text
per-site learnable alpha
```

alpha 数量必须从真实 AttnRes routing sites 自动建立，不按某个模型层数写死。

初始化：

```text
alpha = 0
```

alpha 只控制新的 Block AttnRes routed branch 的强度。

不能把 alpha 放进：

```text
block residual accumulation
source construction
partition selection
block commit
```

Kimi block 内 cumulative residual flow 保持不变。

---

# 10. Identity-preserving conversion

修改完成后增加一个独立 verification test。

对同一输入比较：

```text
original pretrained model
vs.
converted model with alpha = 0
```

在 optimizer step 0 验证 logits。

保存：

```text
identity_test.json
max_abs_logit_diff
mean_abs_logit_diff
base checkpoint hash
converted config hash
```

如果不满足预期数值等价：

```text
STOP
IDENTITY_CONVERSION_FAILED
```

不要通过调 LR、warmup 或别的 heuristic 掩盖。

---

# 11. Query 修改

保留现有正确的 per-site pseudo-query 结构。

需要检查并确保：

```text
site-specific
task-specific
input-independent learned parameter
```

不同任务：

```text
Q_math
Q_multihop
Q_code
...
```

必须独立。

禁止：

```text
Q_math 和 Q_multihop 共用 checkpoint
一个 query bank 配多个 task partition
训练 task B 时沿用 task A optimizer state
只切 partition 不切 query
```

---

# 12. 训练逻辑修改

旧：

```text
freeze backbone
optimizer = query only
```

必须改为：

```text
full-parameter joint post-training
```

对当前 task batch：

```text
update shared backbone
update active Q_t
update active Alpha_t
```

其他 task 的：

```text
Q_other
Alpha_other
```

不更新。

---

# 13. Optimizer 修改

不要把具体 LR 写死在方法代码。

建立至少两个 parameter groups：

```text
Group 1:
pretrained backbone parameters

Group 2:
new AttnRes parameters
    pseudo-query
    alpha
    AttnRes-specific newly introduced parameters
```

方法约束：

```text
AttnRes LR > backbone LR
```

具体 LR 从 experiment config 读取。

不要直接把历史某个模型实验的 LR 写入模型实现。

需要打印并保存：

```text
parameter group names
parameter count
learning rate
weight decay
trainable/frozen status
```

---

# 14. Shared backbone 修改

确认主方法只有一个 shared adapted backbone。

Task-specific 的是：

```text
P_t
Q_t
Alpha_t
```

不要因为旧 query training 是独立 job，就变成：

```text
Math backbone checkpoint
Multi-hop backbone checkpoint
Code backbone checkpoint
```

如果现有训练入口每个 task 都从 base model 独立训练 full backbone，需要修改训练调度，使正式主方法与 SPEC 的 shared-backbone 定义一致。

如现有实验设计对 task training 顺序已有明确约束，则保留该顺序，不自行改变。

---

# 15. Loss 与 token accounting

保留现有正确的 causal LM loss。

如果输入为：

```text
prompt + target
```

则：

```text
prompt/context masked
target supervised
EOS supervised
padding ignored
```

训练预算继续按：

```text
actual non-padding input tokens processed
```

统计，而不是只按 target tokens。

如果旧代码按 answer tokens 计预算，必须修正。

---

# 16. Training verification 修改

增加或修复以下验证。

## 16.1 Backbone

训练前后检查：

```text
backbone parameter delta != 0
```

若不变：

```text
STOP
BACKBONE_NOT_UPDATED
```

---

## 16.2 Query

当前 task：

```text
Q_t changed from initialization
```

---

## 16.3 Alpha

当前 task：

```text
Alpha_t changed from zero
```

如果整个 run 始终为零：

```text
ATTNRES_BRANCH_NEVER_OPENED
```

---

## 16.4 Task isolation

在 task `t` batch 后检查：

```text
Q_t / Alpha_t may change
Q_other / Alpha_other == unchanged
```

---

## 16.5 Partition

训练前后：

```text
partition hash unchanged
```

---

# 17. Probe 修改

如果当前 Probe 逻辑已经正确，不重写。

只验证：

```text
Probe only selects task/mode
Probe does not participate in formal model state
```

Probe 输入不能含：

```text
gold answer
target
dataset name
task token
supporting labels
```

如果 Fixed fallback 仍保留，confidence 必须是实际可拒绝的定义。

不得使用标准二分类：

```text
max softmax < 0.5
```

作为不可达 fallback 条件。

---

# 18. Inference 修改

保留并强化正确流程：

```text
original input
↓
Probe
↓
select (P_t, Q_t, Alpha_t)
↓
discard Probe hidden/cache/state
↓
re-feed original input
↓
start from layer 0
↓
formal generation
```

整个 generation 内配置固定。

禁止：

```text
per-token re-Probe
mid-generation partition switch
mid-generation query switch
cross-case cache reuse
```

---

# 19. Fixed/Kimi baseline 修改边界

Baseline 不从零重建。

只检查其 block residual flow 是否与正式 Kimi-compatible semantics 一致。

Baseline 必须保持：

```text
fixed partition
separate config
separate checkpoint
separate output
separate evaluation record
```

主方法不得把：

```text
P_task
```

写入 Fixed/Kimi baseline。

Baseline 也不得成为 Discovery fallback。

---

# 20. Checkpoint 修改

在现有 checkpoint 机制上补足以下 metadata：

```text
base checkpoint identity/hash
shared backbone identity/hash
enabled tasks

for each task:
    P_t
    partition hash
    Q_t hash
    Alpha_t hash

training config hash
data manifest hash
trained non-padding tokens
```

不要重写整个 checkpoint system，只补当前缺失信息。

---

# 21. Evaluation 修改

保持现有统一 evaluation protocol。

补充记录：

```text
selected partition hash
selected query hash
selected alpha hash/statistics
probe result
selected mode
latency
memory
```

不要因方法修改而改变：

```text
prompt format
answer parser
exact-match rule
evaluation case order
baseline protocol
```

除非当前实现本身已确认错误。

---

# 22. Legacy code 处理

对于已经明确错误、但可能仍被旧脚本引用的逻辑：

```text
true MoiraiBlock replay
query-only trainer
Discovery AttnRes path
```

优先：

```text
disconnect from current entry
mark legacy/deprecated
```

而不是立刻大规模删除文件。

只有确认：

```text
no current import
no current script dependency
no checkpoint compatibility need
```

后再删除。

避免误伤仍然有用的 DP、dataset、evaluation、serialization 代码。

---

# 23. 修改顺序

必须按以下顺序修改：

```text
1. 审计现有代码
2. 标记现有正确/错误模块
3. 修 Discovery 与 AttnRes 的隔离
4. 确认 residual-only Discovery cost 是否真实存在
5. 保留/修复 DP
6. 强化 Kimi-compatible block accumulation
7. 冻结并规范化 P_task 生命周期
8. 增加 per-site alpha
9. 加 identity test
10. 修改 optimizer parameter groups
11. 解冻 shared backbone
12. 实现 task-specific Q/Alpha isolation
13. 修 checkpoint metadata
14. 跑 minimal connectivity test
15. 再运行正式 post-training
16. 检查 Probe/inference
17. 最后运行统一 evaluation
```

不要跳过前面的结构验证直接提交正式训练。

---

# 24. Minimal modification test

正式长训练前，只做最小连通性测试。

验证：

```text
existing pretrained checkpoint can load
Discovery stays in ordinary residual mode
P_task can save/load
Block AttnRes runtime can load P_task
block accumulation matches Kimi semantics
alpha starts at zero
identity test passes
forward/backward completes
backbone receives gradient
alpha receives gradient
query becomes trainable as alpha opens
correct optimizer LR groups load from config
active task bank updates
inactive task banks stay unchanged
checkpoint reload succeeds
```

不要扩展成额外 benchmark。

---

# 25. Codex 修改约束

Codex 必须：

```text
优先修改已有实现
尽量少改文件
复用已经验证代码
每次修改说明旧行为和新行为
先验证再继续下一阶段
所有失败明确报告
```

Codex 不得：

```text
从零重写项目
创建另一套平行主方法逃避旧代码
自行定义 residual Discovery 数学规则
自行新增 AttnRes 变体
修改 Kimi block cumulative residual semantics
为方便实现而改变研究方法
大规模无关重构
静默 fallback
伪造验证通过
```

---

# 26. 修改完成后的验收清单

```text
[ ] 原项目核心目录结构保留
[ ] Discovery 使用原始 pretrained model ordinary residual
[ ] Discovery 不调用 AttnRes/query/alpha
[ ] 旧 true MoiraiBlock replay 不再进入正式 pipeline
[ ] residual-only Discovery cost 已验证，或明确停止等待研究定义
[ ] 原 DP 正确逻辑被保留
[ ] partition 参数由 config/model 决定，不写死模型规模
[ ] 主方法 block 内保持 Kimi cumulative residual flow
[ ] 主方法只改变 task-specific boundaries
[ ] P_task 在训练前冻结
[ ] post-training 不再 rediscover P
[ ] per-site Q_t 存在且 task-specific
[ ] per-site Alpha_t 存在且 zero-init
[ ] step-0 identity test 通过
[ ] backbone 不再冻结
[ ] optimizer 使用 backbone / AttnRes 分组
[ ] LR 从实验 config 读取
[ ] active task Q/Alpha 更新
[ ] inactive task Q/Alpha 不更新
[ ] shared backbone 实际更新
[ ] partition hash 训练前后不变
[ ] Probe state 在正式 inference 前丢弃
[ ] formal inference 从 layer 0 重跑
[ ] Fixed/Kimi baseline 与主方法隔离
[ ] evaluation protocol 未被无关修改
[ ] checkpoint 保存完整 task-specific metadata
```

---

# 27. 最终修改目标

本次修改完成后，现有项目应从旧语义：

```text
pretrained Qwen3
→ Discovery 中混入 AttnRes/compressed replay
→ fixed P
→ freeze backbone
→ query-only training
```

修正为：

```text
pretrained Qwen3
→ ordinary-residual-only task Discovery
→ freeze P_task
→ Kimi-compatible Block AttnRes
→ task-specific Q_t + zero-init Alpha_t
→ identity-preserving conversion
→ shared-backbone full-parameter joint post-training
→ Probe
→ routed inference
```

整个修改过程中必须尽可能保留现有正确实现，而不是重新搭建项目。

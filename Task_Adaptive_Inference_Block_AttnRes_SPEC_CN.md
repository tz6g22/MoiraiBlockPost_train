# Task-Adaptive Inference Block AttnRes 中文完整方法规范（SPEC）

> **文档性质**
>
> 本文件定义 Task-Adaptive Inference Block AttnRes 的当前正式方法语义、阶段边界、禁止事项与验收条件。
>
> 本文件优先于旧版 MoiraiBlock / post-training / Full AttnRes 相关方法说明。
>
> 本文件只固定**方法本身**，不写死任何可能随模型规模或实验设置变化的参数。模型层数、block 数量范围、block 长度范围、样本量、learning rate、batch size、token budget、scheduler、Probe threshold 等全部由 experiment config 决定。

---

# 1. 总流程

主方法严格按照以下顺序执行：

```text
原始 pretrained Transformer
        ↓
Stage 1: ordinary-residual task-specific Discovery
        ↓
得到 P_task
        ↓
冻结 P_task
        ↓
Stage 2: 插入 Kimi-compatible Block AttnRes
        ↓
创建 task-specific pseudo-query
创建 per-site zero-init alpha
        ↓
验证 step-0 identity
        ↓
Stage 3: full-parameter joint post-training
        ↓
shared backbone + task-specific query/alpha
        ↓
Stage 4: Probe training
        ↓
Stage 5: routed inference / evaluation
```

主方法不包含：

```text
Pretrained model
→ Full AttnRes post-training
→ Discovery
→ Task-Adaptive Block AttnRes post-training
```

Full AttnRes 只能作为独立 baseline、upper bound 或 analysis model，不是主方法前置阶段。

---

# 2. 核心对象

对任务 \(t\)，最终配置定义为：

\[
\mathrm{Config}_t=(P_t,Q_t,A_t)
\]

其中：

- \(P_t\)：task-specific partition；
- \(Q_t\)：task-specific pseudo-query bank；
- \(A_t\)：task-specific alpha gate bank。

主方法使用一个共享 backbone：

\[
\theta_{\mathrm{shared}}.
\]

Task-specific 的是：

\[
P_t,\quad Q_t,\quad A_t.
\]

除非额外实验明确改变设置，否则不同任务不拥有各自独立的 full backbone。

---

# 3. 原始 pretrained Transformer

## 3.1 方法起点

整个方法从已经训练完成的标准 pretrained Transformer checkpoint 开始，记原始参数为：

\[
\theta_0.
\]

在 Discovery 开始时必须满足：

```text
模型结构 = 原始 pretrained Transformer
residual connection = 原始模型自带 residual
AttnRes = 不存在
Block AttnRes = 不存在
pseudo-query = 不存在
alpha = 不存在
```

因此 Discovery reference forward 必须是：

\[
f_{\theta_0}^{\mathrm{original}}.
\]

而不是任何 AttnRes 版本。

## 3.2 Ordinary residual 与 AttnRes source 不得混淆

Discovery 可以读取原始模型 forward 中的：

```text
hidden states
residual-stream states
native residual increments
其他经过研究定义允许的原生中间表示
```

这些对象仍然只是原始 pretrained model representation。

不能自动解释成：

```text
AttnRes source
completed block
partial block
Block AttnRes history
```

这些概念只有在 Discovery 完成并正式实例化 Block AttnRes 后才存在。

---

# 4. Kimi Block AttnRes 的严格语义

本方法正式 runtime 的 Block AttnRes 必须继承 Kimi Block AttnRes 的核心计算语义。

本方法**不重新定义新的 block residual architecture**。

## 4.1 Source 语义

正式 Block AttnRes 的 source 必须保持 Kimi-compatible AttnRes 定义。

不得自行把：

```text
Attention output
MLP output
Attention output + MLP output
任意 hidden tensor
```

重新定义为新的 residual source。

Embedding 始终保持独立历史 source。

## 4.2 Block 内部必须是累加式 residual flow

设某连续 block：

\[
B_n=[s_n,e_n].
\]

Block 内 residual contributions 按 Kimi Block AttnRes 语义顺序累加到 current partial state。

抽象表示：

\[
p_n=\sum_{r\in B_n}v_r,
\]

其中 \(v_r\) 表示 Kimi-compatible AttnRes 定义下属于相应 Transformer computation 的 residual contribution。

这里的求和表示：

```text
running residual accumulation
```

不是 pooling。

严格禁止：

```text
mean pooling
average hidden state
learnable pooling
attention pooling
projection summary
MLP summary
weighted averaging
extra compression network
new block encoder
```

## 4.3 Completed block

当运行到当前 task block 的 boundary：

```text
current partial residual state
```

被提交为：

```text
completed block representation
```

即：

\[
p_n\rightarrow b_n.
\]

之后开始下一个 block 的新 partial accumulation。

正式 runtime 的历史状态因此由：

```text
embedding source
+
completed block representations
+
current partial block representation
```

组成。

## 4.4 Partition 单位

Partition 单位始终是**完整 Transformer Block**。

一个完整 Transformer Block 包含：

```text
Attention sublayer
+
MLP sublayer
```

不得出现：

```text
Attention 属于 Block A
MLP 属于 Block B
```

Partition boundary 只能出现在完整 Transformer Blocks 之间。

---

# 5. Kimi Fixed Block 与主方法的关系

设 Kimi-style Fixed Block 使用固定 partition：

\[
P_{\mathrm{fixed}}.
\]

主方法对任务 \(t\) 使用：

\[
P_t.
\]

二者正式 Block AttnRes forward 必须保持相同：

```text
source definition
embedding source semantics
partial-block accumulation
completed-block semantics
routing operator
pseudo-query semantics
source normalization semantics
block commit semantics
```

唯一核心结构差异是：

\[
\boxed{\text{Kimi Fixed Block：固定 boundary}}
\]

而：

\[
\boxed{\text{Task-Adaptive Block：task-specific boundary}}
\]

因此主方法不能额外改变：

```text
block residual rule
block summary operator
source definition
routing operator
block-internal accumulation
```

一句话：

> **主方法保持 Kimi Block AttnRes 的 block 内累加式 residual flow 与 block 间 routing 语义，只把固定 partition boundary 替换成 task-specific partition boundary。**

---

# 6. Stage 1：Task-specific Partition Discovery

## 6.1 输入与输出

对任务 \(t\)，Discovery 输入只有：

\[
(\theta_0,\mathcal D_t^{disc})
\]

其中：

- \(\theta_0\)：原始 pretrained Transformer；
- \(\mathcal D_t^{disc}\)：该任务独立 Discovery 数据。

输出：

\[
P_t.
\]

## 6.2 Discovery 发生在 AttnRes 之前

Discovery 时：

```text
Q_t 尚未建立
A_t 尚未建立
Block AttnRes 尚未实例化
Full AttnRes 不参与
```

因此：

\[
P_t=\operatorname{Discover}
\left(
 f_{\theta_0}^{\mathrm{original}},
 \mathcal D_t^{disc}
\right).
\]

Discovery 必须完全基于原始 pretrained model 的普通 residual / hidden structure。

## 6.3 Discovery 中绝对禁止出现

Discovery 阶段不得：

```text
运行 Full AttnRes
运行 Block AttnRes
运行 Task-Adaptive Block AttnRes
运行 Kimi Fixed Block forward
创建 pseudo-query
读取 pseudo-query
使用 trained query
使用 untrained query
使用 random query
使用 zero query
创建 alpha
读取 alpha
使用 AttnRes softmax
使用 completed AttnRes blocks
使用 partial AttnRes blocks
使用 recency bias
使用 Delta routing
```

---

# 7. Discovery candidate 的语义

Discovery 中候选：

\[
P^{(1)},P^{(2)},\ldots
\]

只表示：

> 对原始 pretrained model 深度结构提出的候选连续分组。

它们不是 runtime Block AttnRes 配置。

因此建议工程上明确区分：

```text
discovery_candidate_partition
```

与：

```text
runtime_partition
```

不能因为得到了候选 partition，就把它送入尚未训练的正式 Block AttnRes forward 来打分。

---

# 8. Discovery representation objective

Discovery 使用 representation-based criterion，不直接使用 generation metric。

对 observation site \(u\)，定义：

- reference representation：\(Z_u^{ref}\)；
- candidate comparison representation：\(Z_u^{cmp}\)；
- valid-token mask：\(\widetilde M\)。

使用 normalized Frobenius distortion：

\[
\delta_u=
\frac{
\left\|
\widetilde M\odot(Z_u^{ref}-Z_u^{cmp})
\right\|_F
}{
\left\|
\widetilde M\odot Z_u^{ref}
\right\|_F+\epsilon
}.
\]

要求：

```text
padding 不参与
norm 使用稳定高精度累积
不同任务独立统计
```

Discovery 主目标不得自行换成：

```text
accuracy
EM
F1
generation score
LM loss
CKA
cosine similarity
```

除非未来 spec 明确修改。

---

# 9. 旧 True MoiraiBlock Replay 明确作废

旧方案曾使用：

```text
local surrogate
→ DP
→ true MoiraiBlock compressed replay
```

其中 true replay 在 query 训练之前直接运行正式 Block AttnRes compressed forward。

该语义已经作废。

原因：

```text
Discovery 时尚未训练 task-specific query
但 old replay 又依赖 AttnRes routing
```

会形成循环：

```text
先用未训练主方法评价 partition
→ 再根据 partition 训练主方法
```

这是方法论错误。

因此以下行为全部禁止：

```text
candidate P → 主方法 Block AttnRes replay
candidate P → Fixed Block replay
candidate P → zero-query replay
candidate P → random-query replay
candidate P → Full AttnRes replay
```

---

# 10. Discovery scoring 的当前研究边界

新版 Discovery scoring 必须是：

```text
ordinary-residual-only
```

即：

\[
C_t(s,e)
\]

以及任何 candidate/global scoring 都只能依赖：

\[
f_{\theta_0}^{\mathrm{original}}
\]

产生的 representation。

如果代码中已经存在经过研究确认的：

```text
ordinary-residual candidate cost
ordinary-residual global scoring
ordinary-residual boundary refinement score
```

则复用。

如果不存在，则必须停止并报告：

```text
RESIDUAL_DISCOVERY_COST_UNDEFINED
```

不得由工程代码自行使用 AttnRes replay 补齐这个研究定义。

原则是：

\[
\boxed{\text{宁可 Discovery 暂时不可执行，也不能用错误 AttnRes 语义把它跑通。}}
\]

---

# 11. Partition Search

一旦合法 residual-only cost 定义完成，即可进行离散 partition search。

设：

\[
P_t=(B_1,\ldots,B_N).
\]

必须满足：

\[
\bigcup_{n=1}^{N}B_n
=
\{0,\ldots,L-1\}
\]

且：

\[
B_i\cap B_j=\varnothing
\quad(i\neq j).
\]

每个：

\[
B_n=[s_n,e_n]
\]

必须连续。

## 11.1 方法级硬约束

以下属于方法结构：

```text
complete Transformer Block as partition unit
continuous
non-overlapping
full coverage
```

如果实验继续保留 `no adjacent singleton`，则把它作为 search constraint 使用。

## 11.2 不得写死的搜索参数

以下全部由 experiment config 决定：

```text
candidate N range
minimum block length
maximum block length
number of Transformer Blocks
```

不得把某个具体模型规模的搜索范围写成方法永久常数。

## 11.3 DP

如果继续使用原 DP，则定义：

```text
DP[i,k,z]
```

其中：

- `i`：已覆盖的完整 Transformer Blocks；
- `k`：当前 block 数；
- `z`：最后一个 block 是否 singleton。

候选长度集合由实验 config 给出。

DP tie-break 必须 deterministic，不得依赖 Python dict/set 遍历顺序。

---

# 12. Task isolation

不同任务必须独立执行 Discovery：

\[
P_t=\operatorname{Discover}(\theta_0,\mathcal D_t^{disc}).
\]

不同任务不得共享：

```text
discovery cases
cost matrix
candidate ranking
DP state
boundary refinement state
final partition
```

如果两个任务最终 partition 相同，只能是独立搜索自然得到相同结果，而不是代码直接复用。

---

# 13. Partition freeze

一旦得到：

\[
P_t,
\]

从正式 post-training 开始即冻结。

禁止：

```text
训练中移动 boundary
训练后 rediscovery
P0 → train → P1
alternating P/train
gradient-based boundary learning
candidate-partition tournament
```

原因是 backbone、query、alpha 会适应当前 partition，训练后再搜索会产生路径依赖。

主方法因此是单向流程：

```text
Discovery
→ freeze P
→ post-training
```

---

# 14. Stage 2：正式插入 Block AttnRes

只有在 \(P_t\) 完全确定并冻结后，才能实例化正式 Block AttnRes。

正确顺序：

```text
Original pretrained model
↓
ordinary-residual Discovery
↓
P_t finalized
↓
instantiate Kimi-compatible Block AttnRes
↓
create Q_t
↓
create A_t
↓
identity test
↓
post-training
```

不得提前插入主方法再进行 Discovery。

---

# 15. Task-specific pseudo-query

对任务 \(t\)：

\[
Q_t=\{q_{t,u}\}_{u\in\mathcal U},
\]

其中 \(\mathcal U\) 由当前模型实际 architecture 与 AttnRes insertion sites 自动决定。

query 数量不得按某个模型规模写死。

## 15.1 Query routing 语义

对当前正式 Block AttnRes sources：

\[
S_1,\ldots,S_K,
\]

产生 key：

\[
k_j=\operatorname{Norm}(S_j).
\]

routing logits：

\[
\ell_j=q^\top k_j.
\]

深度权重：

\[
w_j=\operatorname{softmax}_j(\ell_j).
\]

routed representation：

\[
h^{BAR}=\sum_jw_jS_j.
\]

因此 pseudo-query 的作用是：

> 决定不同历史 block/source 在当前 routing site 的相对使用权重。

## 15.2 Query 本身是静态 learned parameter

Pseudo-query 是：

```text
learned
input-independent
site-specific
task-specific
```

但 routing weights 可以 input-dependent，因为 source key 来自当前输入产生的 representations。

禁止改成：

```text
input-conditioned router
MoE router
dynamic task router
token classifier
```

---

# 16. Alpha：identity-preserving gate

对任务 \(t\)：

\[
A_t=\{\alpha_{t,u}\}_{u\in\mathcal U}.
\]

每个 AttnRes site 有独立 alpha。

初始化：

\[
\alpha_{t,u}=0.
\]

## 16.1 Alpha 的职责

Alpha 只控制：

> 新的 Block AttnRes routed branch 的使用强度。

Alpha 不控制：

```text
partition boundary
block size
source definition
block-internal accumulation
query softmax distribution
```

## 16.2 Alpha 不得进入 Kimi block accumulation

Kimi block 内：

\[
p_n=\sum v_r
\]

必须保持原定义。

禁止：

```text
alpha * v_r 后再累加
alpha 决定 completed block
alpha 决定 block boundary
alpha 充当 block summary
```

## 16.3 Identity-preserving form

设原始 pretrained path 在某 site 的输入为：

\[
h_u^{base},
\]

Block AttnRes routed input 为：

\[
h_u^{BAR}.
\]

采用 identity-preserving 融合：

\[
h_u
=
h_u^{base}
+
\alpha_{t,u}
\left(
h_u^{BAR}-h_u^{base}
\right).
\]

当：

\[
\alpha_{t,u}=0
\]

时：

\[
h_u=h_u^{base}.
\]

因此 converted model 在 step 0 必须保持原始 pretrained behavior。

## 16.4 与 Delta AttnRes 区分

这里的：

\[
h_u^{BAR}-h_u^{base}
\]

只是 identity-preserving correction 的表达方式。

它不代表采用 Delta AttnRes 的 residual source formulation。

本方法仍保持：

```text
Kimi-compatible source semantics
Kimi-compatible cumulative block residual
Kimi-compatible block runtime
```

本方法只借鉴 zero-init learnable gate 的稳定化思想。

---

# 17. Step-0 Identity Test

插入 Block AttnRes 后、optimizer step 1 之前，必须比较：

```text
original pretrained model
vs.
converted model with all alpha = 0
```

要求：

\[
f_{converted}(x)\approx f_{\theta_0}(x).
\]

至少记录：

```text
max_abs_logit_diff
mean_abs_logit_diff
base checkpoint identity/hash
conversion config
```

如果 identity 明显失败：

```text
STOP
IDENTITY_CONVERSION_FAILED
```

不得通过降低 LR、增加 warmup、加 recency bias 等方式掩盖结构错误。

---

# 18. Stage 3：Full-parameter Joint Post-training

正式 post-training 训练：

\[
\theta,\quad Q_t,\quad A_t.
\]

其中：

- \(\theta\)：pretrained backbone；
- \(Q_t\)：当前任务 pseudo-query bank；
- \(A_t\)：当前任务 alpha bank。

旧版：

```text
freeze backbone
train query only
```

已经作废。

新版是：

```text
full-parameter joint post-training
```

---

# 19. Shared backbone 与 task-specific banks

主方法保持一个：

\[
\theta_{shared}.
\]

Task-specific 的是：

\[
(P_t,Q_t,A_t).
\]

当任务 \(t\) 的 batch 到来：

```text
activate P_t
activate Q_t
activate A_t
run forward
compute LM loss
update shared backbone
update Q_t
update A_t
```

其他任务：

```text
Q_other
A_other
```

在该 batch 中不得更新。

---

# 20. Learning-rate 原则

具体 LR 不属于方法常数。

只固定参数组原则：

\[
\eta_{new}>\eta_{backbone}.
\]

其中：

```text
backbone group:
    pretrained model parameters

AttnRes group:
    pseudo-query
    alpha
    其他新引入 AttnRes-specific parameters
```

原因：

- pretrained backbone 已成熟，只需较小更新幅度；
- 新增 routing/gate 参数需要更快适应。

以下全部由实验 config 决定：

```text
backbone LR
AttnRes LR
LR ratio
scheduler
warmup
weight decay
```

不得把某次特定规模实验使用过的绝对数值写成方法永久常量。

---

# 21. Training objective

使用标准 causal LM cross entropy。

对于：

```text
prompt + target
```

loss mask：

```text
prompt/context -> 不监督
target -> 监督
EOS -> 监督
padding -> 忽略
```

本阶段不是：

```text
teacher-student
knowledge distillation
Full AttnRes distillation
hidden-state regression
partition optimization
```

---

# 22. Training budget accounting

即使 loss 只监督 target，训练 compute 仍按：

```text
真实 non-padding input tokens
```

统计。

不能只按 answer tokens 计算训练预算。

至少记录：

```text
optimizer steps
actual non-padding tokens
tokens by task
examples by task
```

---

# 23. Training verification

## 23.1 Backbone 必须实际更新

记录：

```text
backbone state/hash before
backbone state/hash after
parameter delta statistics
```

如果应训练 backbone 但实际未变化：

```text
STOP
BACKBONE_NOT_UPDATED
```

## 23.2 Query 必须更新

每个训练任务：

\[
Q_t\neq Q_t^{init}.
\]

## 23.3 Alpha 必须打开

每个训练任务：

\[
A_t
\]

不能在整个训练期间始终全零。

否则：

```text
ATTNRES_BRANCH_NEVER_OPENED
```

## 23.4 Partition 必须保持不变

```text
partition hash before training
==
partition hash after training
```

## 23.5 Task-bank isolation

训练 task \(t\) 时：

```text
Q_t / A_t may update
Q_other / A_other must not update
shared backbone may update
```

---

# 24. Stage 4：Probe

Probe 唯一职责是：

\[
x\rightarrow t.
\]

然后选择：

\[
(P_t,Q_t,A_t).
\]

Probe 不是：

```text
AttnRes router
layer router
dynamic block selector
token-level router
```

## 24.1 Probe 输入限制

Probe 不得访问：

```text
gold answer
target
dataset name
task token
supporting-fact label
evaluation label
```

只允许 inference-time 可见输入。

---

# 25. Probe 与正式 inference 必须断开

正确流程：

```text
original input
↓
Probe
↓
predicted task / confidence
↓
discard Probe hidden
discard Probe KV/cache
discard Probe state
↓
select Config_t
↓
re-feed original input
↓
formal model starts from layer 0
```

禁止：

```text
直接拿 Probe hidden 继续 generation
复用 Probe cache
从 Probe 结束位置继续正式模型
```

---

# 26. Inference lifecycle

一旦选择任务 \(t\)：

\[
(P_t,Q_t,A_t)
\]

在整个 generation 生命周期内保持固定。

禁止：

```text
每个 token 重新 Probe
中途切换 partition
中途切换 query
中途切换 alpha bank
中途切换 Fixed
跨 case 复用 generation state
```

---

# 27. Fixed mode

如果实验保留 Fixed mode：

\[
\mathrm{Config}_{fixed}
=
(P_{fixed},Q_{fixed},A_{fixed}).
\]

Fixed partition 来自预定义规则，不执行 task-specific Discovery。

Fixed 不得：

```text
进入 task Discovery
作为 Discovery fallback
参与 task DP
参与 candidate ranking
覆盖失败 task checkpoint
```

---

# 28. Independent Kimi / Fixed baseline

Kimi Fixed baseline 与主方法必须代码和输出隔离：

```text
separate config
separate checkpoint
separate output directory
separate evaluation record
```

Baseline 不得读取主方法 task-specific partitions。

主方法不得修改 baseline partition。

比较时应尽量保持：

```text
same data
same token budget
same evaluation protocol
same general post-training principle
```

从而把主要结构变量聚焦于：

\[
P_{fixed}
\quad\text{vs.}\quad
P_t.
\]

---

# 29. Data isolation

至少保持以下集合互斥：

```text
discovery
joint_posttrain_train
joint_posttrain_val
probe_train
probe_val
final_eval
```

使用：

```text
stable_id
content hash
split manifest
```

执行 leakage audit。

Final evaluation 数据不得参与：

```text
Discovery
training
early stopping
Probe training
checkpoint selection
```

---

# 30. 实现与日志中必须区分的四类对象

## A. Original residual state

```text
来自原始 pretrained Transformer
仅用于 Discovery/reference
```

## B. Discovery candidate partition

```text
普通 residual analysis 中的结构候选
不是 runtime Block AttnRes config
```

## C. Kimi-style Block AttnRes runtime state

```text
embedding source
completed blocks
partial block
formal depth routing
```

只在 Discovery 完成后出现。

## D. Task-Adaptive runtime config

```text
P_t + Q_t + A_t
```

用于正式 post-training 和 inference。

不得使用一个模糊的 `block_state` / `compressed_state` / `replay_state` 同时表示上述多种语义。

---

# 31. 明确删除的旧语义

以下全部作废。

## Discovery

```text
Discovery running Full AttnRes
Discovery running Block AttnRes
Discovery running main-method compressed forward
Discovery using Q_full
Discovery using Q_task
Discovery using zero/random query
old true-MoiraiBlock compressed replay
P_fixed as Discovery fallback
```

## Training

```text
freeze entire backbone
query-only optimizer
post-training rediscovery
P0 → train → P1
alternating partition/training
partition tournament
```

## Stabilization

```text
recency bias
Delta residual-source formulation
manual alpha schedule
fixed lambda interpolation schedule
```

除非未来方法 spec 明确重新引入。

---

# 32. Fail-fast 条件

出现以下任何情况必须停止：

```text
Discovery accesses AttnRes parameters
Discovery accesses query/alpha
ordinary-residual Discovery cost undefined
partition violates structural constraints
partition hash changes during training
step-0 identity fails
backbone does not update
wrong task query/alpha bank loads
alpha remains zero for entire run
NaN / Inf loss
data leakage detected
checkpoint/config mismatch
```

不得使用 Fixed mode 或其他 task mode 隐藏失败。

---

# 33. Checkpoint 必须记录的信息

正式 checkpoint 至少记录：

```text
base pretrained checkpoint identity/hash
shared backbone identity/hash
enabled task list

for each task:
    P_t
    partition hash
    Q_t parameter names/hash
    A_t parameter names/hash

training config
optimizer groups
scheduler config
trained non-padding tokens
seed
data manifest hash
```

必须能够明确区分：

```text
P_t
Q_t
A_t
shared backbone
```

避免跨任务或跨 run 混淆。

---

# 34. Evaluation

每个 case 至少记录：

```text
case_id
true task / dataset
probe prediction
probe confidence
selected mode
partition hash
query hash
alpha hash/statistics
answer
correctness
latency
memory
```

总体报告至少包括：

```text
task accuracy / EM / F1
routing accuracy
fallback rate
high-confidence wrong-route rate
end-to-end accuracy
mode-specific accuracy
latency
memory
```

方法分析可额外报告：

```text
task-specific partitions
cross-case partition similarity
cross-task partition similarity
alpha statistics by layer/site
```

---

# 35. 不得写死的实验参数

以下全部必须放到 experiment config，而不是方法实现常数：

```text
model size
model path
number of Transformer Blocks
hidden size
attention heads
candidate N range
block length range
fixed block size
number of discovery examples
number of training examples
training token budget
batch size
gradient accumulation
backbone LR
AttnRes LR
scheduler
warmup
weight decay
Probe threshold
generation max tokens
```

方法代码应从：

```text
model.config
experiment config
dataset manifest
```

读取这些参数。

---

# 36. 当前方法的口头语义

整个方法可以概括为：

> 先拿原始 pretrained Transformer，不改变模型结构，用它自己的普通 residual 表示为不同任务分别做 Discovery，得到每个任务独立的连续 partition。这个阶段完全没有 AttnRes、query 或 alpha，也不能拿候选 partition 去运行主方法 replay。
>
> partition 确定以后立即冻结。随后才把模型转换成 Block AttnRes。正式 Block AttnRes 的 block 内部计算完全保持 Kimi Block AttnRes 的累加式 residual flow：embedding 独立保存，block 内 residual contribution 持续累加，到了 task-specific boundary 后提交为 completed block。主方法相对于 Kimi Fixed Block 的核心结构变化只有 boundary 是 task-specific，而不是固定分块。
>
> 然后为每个任务建立独立 pseudo-query 和 per-site alpha。query 决定不同历史 block/source 的相对 routing 权重；alpha 只控制新的 Block AttnRes 分支使用强度。alpha 从 0 开始，使模型在 step 0 保持原始 pretrained behavior。
>
> 正式 post-training 时 backbone 不冻结，而是 shared backbone、当前任务 query、当前任务 alpha 一起训练。pretrained backbone 使用较小学习率，新加入的 AttnRes 参数使用较大学习率，具体数值由每个实验配置决定。
>
> 最后训练 Probe。Probe 只负责选择任务配置 `(P_t,Q_t,A_t)`。Probe 的 hidden/cache 全部丢弃，再把原始输入从第 0 层重新送入正式模型。在一次 generation 内，选择后的 task configuration 始终保持固定。

---

# 37. 一句话正式定义

> **Task-Adaptive Inference Block AttnRes first discovers task-specific continuous block boundaries solely from the ordinary residual structure of the untouched pretrained Transformer. After these boundaries are frozen, the model instantiates the same cumulative intra-block residual flow and inter-block routing semantics as Kimi Block AttnRes, while replacing fixed block boundaries with task-specific partitions. Task-specific pseudo-queries learn depth routing, and zero-initialized per-site gates provide identity-preserving full-parameter post-training without changing the underlying Kimi block accumulation semantics.**

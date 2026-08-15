# MoiraiBlock Post-Training Tutorial

> **当前有效版本。** 本文档描述 MoiraiBlock 在已训练好的 Hugging Face Qwen3-0.6B 上进行后训练、任务特定 partition discovery、pseudo-query 训练、Probe 路由以及最终推理/评估的完整机制。
>
> 本文档优先于旧版包含 Full AttnRes 预训练、Factual 第三任务、Fixed Block baseline 等描述的计划。

---

# 1. 核心目标

MoiraiBlock 不重新训练一个语言模型，也不先训练一个 Full AttnRes 基座。

起点是已经具备语言能力的 Qwen3-0.6B。MoiraiBlock 只在这个已有模型之上学习**深度信息如何按任务组织和聚合**。

系统最终定义三种并列的推理 mode：

1. **Math mode**
2. **Multi-hop mode**
3. **Fixed Block mode**

最终每个 mode 都由一对不可分割的配置组成：

```text
Math      = (P_math,     Q_math)
Multi-hop = (P_multihop, Q_multihop)
Fixed     = (P_fixed,    Q_fixed)
```

其中：

- `P_*`：Transformer Block partition；
- `Q_*`：该 partition 对应训练得到的 pseudo-query。

三种 mode 在 query training 阶段是**平行独立**的配置。

Fixed Block 只有一个特殊之处：它的 partition 不需要 Discovery，因为它直接采用 Kimi Block AttnRes 的固定分块定义——**每 4 个完整 Transformer Blocks 形成一个 Block**。

Fixed Block **不是 Math/Multi-hop 在 Discovery 或 query training 阶段的 fallback，也不是默认替代结构**。它只在最终推理阶段，当 Probe 对 Math/Multi-hop 的判别置信度低于阈值时，作为第三种备选推理 mode 被选择。

---

# 2. 总流程

```text
                 Hugging Face Qwen3-0.6B
                          │
                          ▼
                Frozen pretrained backbone
                          │
          ┌───────────────┴────────────────┐
          │                                │
     Stage 1A                          Stage 1B
 Math Discovery                  Multi-hop Discovery
          │                                │
          ▼                                ▼
       P_math                         P_multihop
          │                                │
          └───────────────┬────────────────┘
                          │
                          ▼
                   Stage 2 Query Training
                          │
          ┌───────────────┼────────────────┐
          │               │                │
       P_math          P_multihop        P_fixed
          │               │                │
       train Q          train Q          train Q
          │               │                │
       Q_math          Q_multihop        Q_fixed
          │               │                │
          └───────────────┼────────────────┘
                          │
                          ▼
                   Stage 3 Probe Training
                    Math vs Multi-hop
                          │
                          ▼
                 Stage 4 Inference / Eval
                          │
                  Probe + confidence
               ┌──────────┼───────────┐
               │          │           │
             Math     Multi-hop   low confidence
               │          │           │
        P_math,Q_math  P_multi,Q_multi P_fixed,Q_fixed
```

---

# 3. AttnRes / Block 表示的统一语义

## 3.1 Partition 的搜索单位

MoiraiBlock 的 partition 单位始终是**完整 Transformer Block**。

一个完整 Transformer Block 包含：

```text
Attention sublayer
+
MLP sublayer
```

不能把 Attention 和 MLP 拆到两个不同的 MoiraiBlocks 中。

因此一个候选 MoiraiBlock 必须是连续 Transformer Blocks：

\[
B(s,e)=\{s,s+1,\ldots,e\}
\]

其中 `s` 和 `e` 都是完整 Transformer Block 的索引。

必须满足：

```text
continuous = true
non_overlapping = true
full_coverage = true
```

---

## 3.2 Residual source 语义

MoiraiBlock 不重新定义 Kimi AttnRes 的 residual source。

Embedding 保持独立 source；每个 Transformer Block 内部的 Attention residual contribution 和 MLP residual contribution 仍按照 AttnRes 原有语义产生。

Partition 只是决定：

> 哪些连续 Transformer Blocks 的 residual contributions 在 block boundary 被累计成一个 block-level representation。

对于连续区间 `B(s,e)`，其 block representation 可抽象表示为：

\[
b_{s,e}=\sum_{r=s}^{e} v_r
\]

这里的 `v_r` 表示属于第 `r` 个完整 Transformer Block 的 residual contribution 集合在 Block AttnRes 语义下的累计贡献。

禁止增加：

- learnable block summary；
- mean pooling；
- projection；
- normalization-only summary；
- 额外 gating；
- 非连续 source 聚合。

Block summary 必须保持 Kimi Block AttnRes 的 residual accumulation 语义。

---

# 4. Stage 1：Task-specific Partition Discovery

Stage 1 只对两个任务执行：

```text
Math
Multi-hop
```

Fixed Block **完全跳过 Discovery**。

原因不是 fallback，而是：

```text
P_fixed = predefined fixed partition
每个 block 固定包含 4 个 Transformer Blocks
```

因此 Discovery 的输出只有：

```text
P_math
P_multihop
```

---

# 5. Discovery 的总体算法

对 Math 和 Multi-hop 分别独立执行：

```text
Frozen reference forward
        ↓
构建 candidate interval local cost
        ↓
DP 搜索每个候选 block 数 N 的最优 partition
        ↓
对每个 P_N 做真实 compressed replay
        ↓
选择 N*
        ↓
boundary refinement
        ↓
最终 P_task
```

Math 和 Multi-hop 从数据、cost matrix、DP、replay 到最终 partition 都必须独立。

Fixed partition 不能参与任何一步。

---

# 6. Discovery Observation Sites

为了衡量一个 partition 对原模型深度信息流造成的扰动，Discovery 不使用 task accuracy 直接搜索，而使用内部表示 distortion。

Observation site 放在 AttnRes 聚合完成、进入实际子层计算之前。

对于第 `r` 个 Transformer Block：

- `z_r^A`：进入 Attention 前的聚合表示；
- `z_r^M`：进入 MLP 前的聚合表示；
- `z^F`：最后一个 Transformer Block 完成后、最终归一化之前的表示。

记所有 observation sites 为：

\[
\mathcal U
\]

这些位置用于比较：

```text
reference execution
vs.
compressed/block execution
```

---

# 7. Distortion 定义

设某个 observation site `u`：

- reference hidden：`Z_u^ref`
- compressed hidden：`Z_u^cmp`
- 有效 token mask：`M`

使用 masked normalized Frobenius error：

\[
\delta_u =
\frac{
\|\widetilde M \odot (Z_u^{ref}-Z_u^{cmp})\|_F
}{
\|\widetilde M \odot Z_u^{ref}\|_F + \epsilon
}
\]

要求：

- padding token 不参与；
- norm 使用稳定的高精度累积；
- 不用 task accuracy 代替 hidden distortion；
- 不用 cosine/CKA 代替当前 DP 主目标；
- 不把 generation metric 放进 partition 搜索目标。

---

# 8. Candidate Interval Local Surrogate Cost

DP 需要可加的区间 cost，因此不能对每种完整 partition 都直接穷举真实 forward。

对于候选连续区间：

\[
B(s,e)
\]

只允许区间长度：

\[
1 \le e-s+1 \le 4
\]

## 8.1 单个 discovery case

对一个 case：

1. 运行一次冻结 reference forward；
2. 保存计算区间 cost 必要的 residual sources 和 reference observation states；
3. 对候选区间 `[s,e]`，把该区间内的独立 residual contributions 替换为一个 block representation；
4. 区间之外的 reference source tensor 保持不变；
5. 只重新计算该区间之后的 depth/AttnRes aggregation；
6. **不重新执行下游 Attention 和 MLP**。

因此这个 cost 是一个用于 DP 搜索的**局部 surrogate**。

它不是最终真实 compressed forward。

---

## 8.2 下游 observation sites

对于 `[s,e]`，只比较位于该区间之后的 observation sites：

\[
\mathcal U_{>e}
\]

单 case local cost：

\[
C_x(s,e)=
\frac{1}{|\mathcal U_{>e}|}
\sum_{u\in\mathcal U_{>e}}
\delta_{x,u}^{[s,e]}
\]

任务级 cost：

\[
C_{task}(s,e)=
\frac{1}{|\mathcal D_{discovery}|}
\sum_x C_x(s,e)
\]

最终形成一个 task-specific cost matrix：

```text
C_math(s,e)
C_multihop(s,e)
```

两者绝不能共用。

---

# 9. 为什么 Local Cost 之后还需要 Full Replay

DP 优化的是：

\[
\sum_j C_{task}(B_j)
\]

但多个 block 同时压缩后会相互影响，所以一般：

\[
\sum_j C_{task}(B_j)
\neq
D_{task}(P)
\]

其中 `D_task(P)` 是整个 partition 真正跑一遍 compressed model 后得到的全局 distortion。

因此：

> Local surrogate 只负责高效搜索；最终 partition 的比较必须通过真实 full replay 完成。

这是 Discovery 算法的重要边界。

---

# 10. DP Partition Search

## 10.1 状态

定义：

\[
DP[i,k,z]
\]

表示：

> 将前 `i` 个完整 Transformer Blocks 划分为 `k` 个 MoiraiBlocks 时的最小累计 surrogate cost。

其中 `z` 表示最后一个 block 是否为 singleton：

\[
z=
\begin{cases}
1,& \text{最后一个 block 长度为 1}\\
0,& \text{最后一个 block 长度大于 1}
\end{cases}
\]

初始化：

\[
DP[0,0,0]=0
\]

其他状态为 `+∞`。

---

## 10.2 候选 block 长度

每个 MoiraiBlock 长度只能为：

```text
1, 2, 3, 4
```

即：

\[
m\in\{1,2,3,4\}
\]

当前最后一个 block：

\[
[i-m, i-1]
\]

---

## 10.3 禁止相邻 singleton

如果前一个 block 长度为 1，当前候选 block 也为 1，则该转移非法。

```python
if z_prev == 1 and m == 1:
    continue
```

这个约束防止搜索产生连续的碎片化单层 blocks。

---

## 10.4 DP 转移

令：

\[
z_{new}=\mathbf 1[m=1]
\]

合法转移：

\[
DP[i,k,z_{new}]
=
\min_{m,z_{prev}}
\left(
DP[i-m,k-1,z_{prev}]
+
C_{task}(i-m,i-1)
\right)
\]

同时保存 predecessor：

```text
prev_i
prev_k
prev_z
chosen_length
```

最终通过 backtracking 精确恢复 partition。

---

# 11. DP Partition 必须满足的硬约束

每个 DP 输出必须满足：

```text
所有 Transformer Blocks 被完整覆盖
blocks 连续
blocks 不重叠
block length ∈ [1,4]
不存在相邻 singleton
block_count == N
```

任何违反约束的 partition 都不是合法 MoiraiBlock partition。

---

# 12. 候选 Block 数 N

当前 Discovery 对多个候选 `N` 分别进行 DP 搜索，而不是提前写死一个 MoiraiBlock 数量。

当前规则搜索：

```text
N = 9 ... 16
```

每个 `N` 得到一个：

```text
P_N
```

注意：

- `P_fixed` 不在这个搜索中；
- Fixed Block 不作为 DP 初始 partition；
- Fixed Block 不作为 DP fallback；
- Fixed Block 不参与 N selection。

---

# 13. DP Tie-break

为了保证相同输入得到完全确定的 partition，当两个 DP path 的 cost 在数值容差内相同时，必须使用确定性的 tie-break。

顺序：

1. 优先更早出现的 boundary；
2. 若仍相同，优先当前 block 长度较大者；
3. 若仍相同，选择字典序更小的 block-length sequence。

不能依赖 Python dict/set 遍历顺序决定最终 partition。

---

# 14. True Full Partition Replay

DP 得到的 `P_N` 只是 surrogate objective 下的最优候选。

对每个 `P_N`，必须运行**真实 MoiraiBlock compressed forward**：

```text
input
 ↓
按照 P_N 维护 completed block representations
 ↓
block 内维护当前 partial residual accumulation
 ↓
Attention / MLP 都真实重新计算
 ↓
后续 hidden 真实依赖前面已经 compressed 的结果
```

不能继续复用 local surrogate 的缓存来冒充完整 replay。

对一个 case 的全局 distortion：

\[
D_x(P)=
\frac{1}{|\mathcal U|}
\sum_{u\in\mathcal U}
\delta_{x,u}^{P}
\]

任务平均：

\[
D_{task}(P)=
\frac{1}{|\mathcal D_{discovery}|}
\sum_x D_x(P)
\]

---

# 15. N Selection

对每个候选 `N`：

```text
DP → P_N
P_N → true replay → D_task(P_N)
```

首先找到：

\[
D_{min}=\min_N D_{task}(P_N)
\]

然后构造“接近最优”的候选集合：

\[
D_{task}(P_N) \le D_{min}+\tau
\]

其中当前规则：

\[
\tau=\max(0.02D_{min},10^{-8})
\]

在这些接近最低 distortion 的候选中，选择**最小的 N**：

\[
N^*=\min\mathcal N_{ok}
\]

目的：

> 在 distortion 基本等价时，优先选择 block 数更少、压缩更强的 partition。

得到初始 partition：

\[
P^{(0)}=P_{N^*}
\]

---

# 16. Boundary Refinement

DP 的 boundary 由局部 surrogate cost 决定，因此 N 选定后还要使用真实 replay 做局部边界优化。

保持：

```text
N = N*
```

不变。

对每一条内部 boundary：

```text
尝试向左移动 1 个 Transformer Block
尝试向右移动 1 个 Transformer Block
```

每个候选都重新检查：

```text
block length 1..4
no adjacent singleton
continuous
full coverage
```

合法候选执行完整 true replay。

只有新 partition 的真实 distortion 满足：

\[
D_{old}-D_{new}
>
\max(10^{-8},10^{-6}D_{old})
\]

才接受移动。

如果一整轮没有有效改进，则停止；否则继续有限轮 refinement。

最终输出：

```text
P_math
P_multihop
```

---

# 17. Fixed Block Partition

Fixed mode 不执行上述 Discovery。

直接按照 Kimi Block AttnRes 的固定规则：

```text
每 4 个完整 Transformer Blocks → 一个 Block
```

得到：

```text
P_fixed
```

对于当前 Qwen3-0.6B，这个规则直接按模型真实 Transformer Block 顺序连续分组。

必须强调：

```text
P_fixed 不进入 Math DP
P_fixed 不进入 Multi-hop DP
P_fixed 不作为 local surrogate 的默认 partition
P_fixed 不作为 replay 的 fallback
P_fixed 不参与 N selection
P_fixed 不参与 boundary refinement
```

Discovery 与 Fixed mode 之间没有“托底”关系。

---

# 18. Stage 2：Pseudo-query Training

Discovery 完成后，有三套 partition：

```text
P_math       ← Discovery
P_multihop   ← Discovery
P_fixed      ← predefined fixed-size partition
```

接下来为三套 partition **分别训练自己的 pseudo-query**。

---

# 19. Frozen Backbone

Query training 时冻结 Qwen3 backbone。

冻结至少包括：

```text
embedding
Attention weights
MLP weights
normalization parameters
LM head
所有非 pseudo-query 参数
```

只有当前 mode 对应的 pseudo-query 可以加入 optimizer。

梯度仍然需要穿过冻结 backbone 的计算图传播到 pseudo-query；“冻结 backbone”不等于 `no_grad()` 包住整个 forward。

---

# 20. 三条平行 Query Training 路径

## 20.1 Math

```text
P_math
+
frozen Qwen3
+
Math training data
        ↓
train pseudo-query only
        ↓
Q_math
```

最终：

```text
Config_math = (P_math, Q_math)
```

---

## 20.2 Multi-hop

```text
P_multihop
+
frozen Qwen3
+
Multi-hop training data
        ↓
train pseudo-query only
        ↓
Q_multihop
```

最终：

```text
Config_multihop = (P_multihop, Q_multihop)
```

---

## 20.3 Fixed

```text
P_fixed
+
frozen Qwen3
+
Fixed-mode query training data
        ↓
train pseudo-query only
        ↓
Q_fixed
```

最终：

```text
Config_fixed = (P_fixed, Q_fixed)
```

`P_fixed` 不需要 Discovery，但 `Q_fixed` 必须经过 query training，否则 Fixed mode 没有完整的可用 AttnRes aggregation 参数。

三条 query training 路径在这一阶段是**平行关系**。

Fixed mode 此时仍然不是 Math 或 Multi-hop 的 fallback。

---

# 21. Query Training Loss

每个 mode 使用其对应训练文本执行 causal language modeling。

核心 forward：

```text
input
 ↓
fixed partition P_mode
 ↓
Block AttnRes aggregation with Q_mode
 ↓
frozen Transformer computation
 ↓
LM logits
```

只更新 `Q_mode`。

目标使用真实语言模型 CE loss；如果训练样本由 prompt + target 构成，则 loss 只在 target/EOS 位置计算，prompt 与 padding 不进入监督 loss。

这不是：

- teacher-student；
- knowledge distillation；
- hidden-state regression；
- backbone fine-tuning。

---

# 22. Query Training 的独立性约束

必须最终保存三套独立 query：

```text
Q_math
Q_multihop
Q_fixed
```

禁止：

```text
Q_math == Q_multihop checkpoint
Q_fixed 覆盖 Q_math
Q_multihop 从 Q_math optimizer state 继续训练
一个 query 在推理时搭配多个不同 partition
只切换 partition 不切换 query
```

每个 mode 的 partition 与 query 必须作为一个 bundle 使用。

---

# 23. Query Training 的计算量计数规则

Query training 虽然只更新 pseudo-query，但 forward/backward 仍然经过完整冻结 backbone，因此实际计算量主要由**完整 input token 数**决定。

训练预算必须按：

```text
non-padding input tokens
```

计数，而不是只按 answer/target token 数计数。

原因：Multi-hop 输入可能很长而答案很短；若只计算 target tokens，会让 Multi-hop 在相同 nominal token budget 下实际执行远多于 Math 的 Transformer token 计算。

因此：

> loss mask 可以只监督 target；训练计算预算必须按真实 non-padding input tokens 计数。

---

# 24. Stage 3：Probe Classifier

Probe 只负责一个决策：

> 当前输入是否能被高置信度识别为 Math 或 Multi-hop？

Probe 训练标签只有：

```text
Math
Multi-hop
```

Fixed 不是一个训练任务类别；Fixed 是**低置信度情况下的推理 mode**。

---

# 25. Probe 推理决策

对于输入 `x`，Probe 输出：

```text
predicted_task
confidence
```

决策规则：

```text
if confidence >= 0.5 and predicted_task == math:
    mode = math

elif confidence >= 0.5 and predicted_task == multihop:
    mode = multihop

else:
    mode = fixed
```

即：

```text
high-confidence Math
→ (P_math, Q_math)

high-confidence Multi-hop
→ (P_multihop, Q_multihop)

confidence < 0.5
→ (P_fixed, Q_fixed)
```

---

# 26. Probe Confidence 的实现要求

这里有一个必须避免的实现陷阱。

如果把 confidence 定义为标准**二分类 softmax 的最大概率**：

\[
\max(p_{math},p_{multihop})
\]

那么它理论上总是 `>= 0.5`，低于 `0.5` 的 fallback 几乎永远不会发生。

因此实现必须保证项目中的 `confidence` 是一个**确实可以低于 0.5 的置信度量**，例如使用能够表达“不属于两个已知分布”的校准 confidence/acceptance score。

本文档的硬功能要求不是限定某一种校准算法，而是：

```text
confidence < 0.5 必须在实际系统中可达
```

否则 Fixed fallback 分支在数学上就是死代码，违反设计意图。

---

# 27. Probe 只负责选择配置，不参与正式推理状态

Probe forward 的 hidden/state 不能直接继续用于正式 generation。

正确流程：

```text
original input
      ↓
shallow Probe forward
      ↓
predicted_task + confidence
      ↓
discard Probe hidden/cache/state
      ↓
select Config_mode
      ↓
original input again
      ↓
restart from Transformer Block 0
      ↓
formal inference
```

原因：Probe 只是配置选择器，不应该消耗或替代正式模型层。

因此必须保证：

```text
Probe hidden discarded
Probe KV/cache discarded
formal forward starts from layer 0
original tokens are re-fed
```

---

# 28. Stage 4：Inference

一个 case 的最终推理流程：

```text
Case input
   ↓
Probe
   ↓
predicted task + confidence
   ↓
┌───────────────────────────────┐
│ confidence >= 0.5 ?           │
└──────────────┬────────────────┘
               │
       yes     │      no
        │      │       │
        ▼      │       ▼
 Math / Multi-hop      Fixed
        │              │
        ▼              ▼
load task P+Q      load P_fixed+Q_fixed
        │              │
        └───────┬──────┘
                ▼
        restart original input
        from Transformer Block 0
                ↓
          complete generation
```

---

# 29. Math Mode

若 Probe 高置信度判定为 Math：

```text
active_partition = P_math
active_query     = Q_math
active_mode      = math
```

正式 generation 的整个生命周期内必须保持这一配置不变。

不能出现：

```text
P_math + Q_multihop
P_math + Q_fixed
中途切回 Fixed
每个 generated token 重新 Probe
```

---

# 30. Multi-hop Mode

若 Probe 高置信度判定为 Multi-hop：

```text
active_partition = P_multihop
active_query     = Q_multihop
active_mode      = multihop
```

完整 generation 内保持不变。

---

# 31. Fixed Mode

只有最终推理阶段出现以下条件时：

```text
Probe confidence < 0.5
```

才选择：

```text
active_partition = P_fixed
active_query     = Q_fixed
active_mode      = fixed
```

这里的 Fixed 是：

> 对“Probe 无法高置信度归入 Math 或 Multi-hop”的输入采用的预定义 Block AttnRes partition 方案。

必须再次强调：

```text
Fixed 不参与 Discovery
Fixed 不给 Math Discovery 托底
Fixed 不给 Multi-hop Discovery 托底
Fixed 不给 Math query training 托底
Fixed 不给 Multi-hop query training 托底
Fixed 不覆盖失败的 task checkpoint
```

它只在 inference routing 的低置信度分支被选择。

---

# 32. 连续不同任务输入时的状态切换

系统必须支持逐 case 切换完整 `(P,Q)` bundle。

例如：

```text
Case 1: Math
→ P_math + Q_math

Case 2: Multi-hop
→ P_multihop + Q_multihop

Case 3: low confidence
→ P_fixed + Q_fixed

Case 4: Math
→ P_math + Q_math
```

不能让前一条输入残留的：

- active partition；
- active query；
- Probe cache；
- generation KV cache；

污染下一条输入。

---

# 33. Evaluation

最终 Evaluation 评估的是**整个 routing + mode-specific configuration 系统**，而不是只评估 Probe 或只评估某一个 partition。

每条 case 至少记录：

```text
true task / dataset
probe prediction
probe confidence
selected mode
selected partition hash
selected query hash
answer correctness
latency
memory / efficiency
```

对于 MoiraiBlock 的核心实验，还应保留 representation distortion，用于分析 task-specific partition 相对于 reference 深度信息流的保持程度。

建议同时统计：

```text
Math routed to Math rate
Multi-hop routed to Multi-hop rate
Fixed selection rate
high-confidence wrong-route rate
end-to-end task accuracy
mode-specific accuracy
latency
memory
```

---

# 34. 三种 Mode 的关系总结

三种模式在 query training 阶段：

```text
Math            Multi-hop          Fixed
 │                  │                │
P_math          P_multihop        P_fixed
 │                  │                │
Q_math          Q_multihop        Q_fixed
```

是平行关系。

唯一差异是 partition 来源：

```text
P_math      ← task-specific Discovery
P_multihop  ← task-specific Discovery
P_fixed     ← predefined 4-Transformer-Block grouping
```

只有到了 inference：

```text
Probe high confidence Math      → Math mode
Probe high confidence Multi-hop → Multi-hop mode
Probe low confidence            → Fixed mode
```

所以 Fixed 的“备选”身份**只存在于最后的推理选择阶段**。

---

# 35. 实现中必须禁止的错误语义

以下任何一种实现都属于流程错误：

## Discovery 错误

```text
Discovery 默认 mode=fixed
Math DP 从 P_fixed 开始
Multi-hop DP 从 P_fixed 开始
Discovery 失败后返回 P_fixed
P_fixed 参与 N selection
P_fixed 参与 boundary refinement
Math/Multi-hop 共用 cost matrix
Math/Multi-hop 共用最终 partition
```

## Query Training 错误

```text
Math trainer 名义加载 P_math，底层 forward 实际 mode=fixed
Multi-hop trainer 名义加载 P_multihop，底层实际使用 P_fixed
Q_fixed 被 Math/Multi-hop 共用
Q_math 与 Q_multihop 保存到同一 checkpoint
backbone 被 optimizer 更新
```

## Probe 错误

```text
Fixed 被当成第三个 supervised task label
Probe 的 max two-class softmax probability < 0.5 作为 fallback 条件
Probe hidden 继续进入正式 forward
Probe KV cache 继续用于 generation
```

## Inference 错误

```text
Probe=Math 但只切换 partition、不切 query
Probe=Multi-hop 但仍使用 Math query
low confidence 仍强制进入 Math/Multi-hop
Fixed 在高置信度 Math/Multi-hop 时覆盖 task mode
下一 case 沿用上一 case 的 partition/query
```

---

# 36. 最终不可变的功能闭环

最终 MoiraiBlock 后训练必须形成以下闭环：

```text
1. 已训练 Qwen3-0.6B

2. Math Discovery
   → P_math

3. Multi-hop Discovery
   → P_multihop

4. Fixed partition
   → P_fixed = 每4个完整 Transformer Blocks 一组

5. Frozen-backbone query training
   P_math      → Q_math
   P_multihop  → Q_multihop
   P_fixed     → Q_fixed

6. Probe training
   Math / Multi-hop classification + usable confidence

7. Inference
   high-conf Math      → P_math + Q_math
   high-conf Multi-hop → P_multihop + Q_multihop
   low-conf            → P_fixed + Q_fixed

8. Probe state discarded
   original input restarts from Transformer Block 0

9. Evaluation
   evaluate final selected mode and generated result
```

这才是当前 MoiraiBlock post-training 设计。

---

# 37. 一句话概括

> **MoiraiBlock 先为 Math 和 Multi-hop 分别发现最适合其深度信息流的连续非均匀 partition，再在冻结 Qwen3 backbone 下为 Math、Multi-hop 和固定 4-layer Block 三种 partition 分别训练独立 pseudo-query；推理时 Probe 高置信度选择对应任务的 `(partition, query)`，低置信度则选择独立训练好的 Fixed Block `(P_fixed,Q_fixed)`，随后丢弃 Probe 状态并从 Transformer Block 0 重新执行正式推理。**

# MoiraiBlock Qwen3-14B 后训练

当前唯一方法规范是 [`tutorial_postTraining.md`](tutorial_postTraining.md)。

项目从 Hugging Face `Qwen/Qwen3-14B` 转换得到保留原始 backbone
权重的 AttnRes 执行模型，随后直接运行：

```text
Math / Multi-hop / Code Partition Discovery
→ Math / Multi-hop / Code / Fixed Query Training
→ Math / Multi-hop / Code Probe Training
→ Probe Routing Inference / Evaluation
```

不存在 Full AttnRes 300M 前置训练。Fixed partition 每四个完整 Transformer
Blocks 分为一组，不参与 Discovery。

当前数据数量：

- Discovery：Math 1000（GSM8K 500 + SVAMP 500），Multi-hop 1000，Code 200（MBPP pool）。
- Query training：Math 1000，Multi-hop 1000，Code 200（MBPP pool）。
- Fixed query training：Math 1000 + Multi-hop 1000 + Code 200，共 2200。
- Probe training：Math/Multi-hop 各 200，Code 100。
- 最终评估：每个任务 10 case，以 accuracy 为主要指标；Code 使用 MBPP 官方测试断言。

Discovery 和 Query Training 各阶段内部均使用 unique examples；两阶段之间允许
复用同一 stable ID。官方 train/validation/test 只作为 metadata，数据按统一 pool
分配；最终 evaluation 只使用所有前序用途均未出现过的 stable ID。

## 运行

```bash
scripts/setup_venv.sh
scripts/prepare_assets.sh
NPROC_PER_NODE=4 scripts/run_pipeline.sh
```

14B 模型阶段使用 `torchrun`、PyTorch FSDP `FULL_SHARD` 和 BF16；自动
wrap 单元是 Qwen3 Transformer decoder layer。无需保持终端连接的 IRIDIS
后台入口为：

```bash
scripts/run_iridis_4xl4_background.sh
```

只检查命令：

```bash
scripts/run_pipeline.sh --dry-run
```

正式输出位于 `outputs/formal/`，下载数据和缓存位于 `artifacts/`。

## 验证

```bash
.venv/bin/python -m pytest -q
```

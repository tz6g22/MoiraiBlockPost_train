# MoiraiBlock Qwen3-0.6B 后训练

当前唯一方法规范是 [`tutorial_postTraining.md`](tutorial_postTraining.md)。

项目从 Hugging Face `Qwen/Qwen3-0.6B` 转换得到保留原始 backbone
权重的 AttnRes 执行模型，随后直接运行：

```text
Math / Multi-hop Partition Discovery
→ Math / Multi-hop / Fixed Query Training
→ Math-vs-Multi-hop Probe Training
→ Probe Routing Inference / Evaluation
```

不存在 Full AttnRes 300M 前置训练。Fixed partition 每四个完整 Transformer
Blocks 分为一组，不参与 Discovery。

当前数据数量：

- Discovery：Math 200（GSM8K 100 + SVAMP 100），Multi-hop 200。
- Query training：Math 200，Multi-hop 200。
- Fixed query training：Math 100 + Multi-hop 100。
- Probe training：每类 200。
- 最终评估：每个任务 10 case，以 accuracy 为主要指标。

## 运行

```bash
scripts/setup_venv.sh
scripts/prepare_assets.sh
scripts/run_pipeline.sh
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

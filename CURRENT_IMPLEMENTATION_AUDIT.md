# Current Implementation Audit

This audit records the implementation state before the formal-method changes.

| Area | Current implementation | Status against the formal SPEC |
| --- | --- | --- |
| Discovery entry | `src/discovery/run_all.py` | Needs change |
| Discovery model | Loads `MoiraiQwen3ForCausalLM` from `qwen3_14b_full_attnres` | Wrong: this is a converted Full AttnRes model, not the original Qwen3 model |
| Discovery cost | `collect_full_reference` + `local_surrogate_interval_cost` | Wrong for the formal path: it reads AttnRes observations |
| Candidate scoring | `replay_partition` in `_finish_task_discovery` | Wrong: formal Discovery must not score candidates with Block AttnRes replay |
| DP / serialization | `solve_partition`, `MoiraiPartition`, JSON/hash serialization | Reusable, subject to a residual-only cost definition |
| Block flow | `src/modeling/full_attnres.py` and `src/modeling/block_attnres.py` | Cumulative partial/completed source sums are reusable; alpha and identity fusion are missing |
| Query | `attn_pseudo_query`, `mlp_pseudo_query`, `final_pseudo_query` | Reusable as site-specific parameters; formal task banks still need explicit alpha and task isolation |
| Backbone training | `src/adapter/train_query.py::freeze_for_query_training` | Wrong: freezes every non-query parameter and builds a query-only optimizer |
| Optimizer | Query-only parameter list in `train_query_partition` | Wrong: formal training needs shared backbone and active task Q/Alpha groups |
| Probe | `src/probe/inference.py` and `src/evaluation/run_evaluation.py` | Wrong: Fixed is still an allowed bundle and low-confidence fallback |
| Inference lifecycle | Probe then formal generation from copied original input | Refeed behavior is present, but Fixed routing must be removed |
| Evaluation | `src/evaluation/run_evaluation.py` | Wrong: evaluates Full/Fixed baselines and loads a Fixed bundle |
| Checkpoint | `MoiraiConfigBundle` validates query-only manifests | Needs extension for task-specific alpha and shared-backbone metadata |

The formal residual-only Discovery cost is not defined by the current repository.
The legacy cost path cannot be silently reused; the formal Discovery entry therefore
fails with `RESIDUAL_DISCOVERY_COST_UNDEFINED` until that research definition is
provided.

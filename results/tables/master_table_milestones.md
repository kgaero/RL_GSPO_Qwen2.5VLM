# Master Table A: Milestones Only

CSV source: `tables/master_table_milestones.csv`

## Purpose

Compact report-ready summary of the baseline, major phase milestones, important regressions, and final checkpoint recommendations.

## Column Glossary

### Identity and Selection

| Column | Simple meaning |
| --- | --- |
| `Milestone ID` | Stable short name used to refer to a key row in the report. |
| `Run Family` | Internal run bucket showing which notebook family produced the row. |
| `Run Family Label` | Human-friendly label for the run family. |
| `Notebook / Run Slug` | Kaggle or local run identifier that produced the artifacts. |
| `Output Root` | Top-level output folder where this run wrote checkpoints and reports. |
| `Phase` | Training phase name used by the staged RL pipeline. |
| `Phase Label` | Human-friendly phase name used in tables and plots. |
| `Checkpoint` | Checkpoint directory name. |
| `Global Step` | Trainer step number at which the checkpoint was saved. |
| `Timeline Order` | Chronological order used in the report figures. |
| `X Label` | Short checkpoint label shown on the x-axis of plots. |
| `Artifact Status` | Whether this row came from a real checkpoint, alias metadata, or a baseline snapshot. |
| `Metrics Source Kind` | Which file type supplied the metrics for the row. |
| `Alias Role` | Primary alias assigned to the checkpoint, such as best composite or latest. |
| `Alias Roles` | All alias labels that point to the checkpoint. |
| `Keep / Discard` | Report-level recommendation for whether to keep this checkpoint for future use. |
| `Decision Reason` | Short explanation for the keep/discard recommendation. |
| `Key Interpretation` | Plain-language take on what the row means. |

### Training Setup

| Column | Simple meaning |
| --- | --- |
| `Train Split` | Dataset split used for training in this run. |
| `Eval Split` | Dataset split used for checkpoint evaluation. |
| `Hardware Profile` | Named runtime profile applied to fit the hardware budget. |
| `Seed / Resume Source` | Resume selector used to continue from a previous checkpoint alias. |
| `Warm-Start Checkpoint` | Explicit checkpoint path loaded before training started. |
| `Training Strategy Introduced` | Short summary of what this phase was trying to teach the model. |
| `Stage Mix / Curriculum` | Curriculum subset mix used in the phase. |
| `Phase Description` | Longer phase description from the run config. |
| `Phase Default Resume Selector` | Default alias that this phase expects to resume from. |
| `Base Model` | Base pretrained model used before LoRA adaptation. |
| `4-bit Enabled` | Whether 4-bit loading was used to reduce memory. |
| `Fast Inference Enabled` | Whether the fast generation path was enabled during RL. |
| `Fast Inference Mode` | Short name for the inference backend or mode. |
| `Compilation Mode` | vLLM/Unsloth compilation setting used for stable generation. |
| `max_seq_length` | Maximum total sequence length allowed by the runtime profile. |
| `image_size` | Target image resolution used for visual inputs. |
| `LoRA Rank` | Adapter rank used for LoRA fine-tuning. |
| `Max LoRA Rank` | Upper LoRA rank bound passed to the runtime when supported. |
| `gradient_accumulation_steps` | How many micro-batches were accumulated before an optimizer step. |
| `gpu_memory_utilization` | Target GPU memory fraction reserved for the fast generation backend. |
| `num_generations` | How many completions were sampled per prompt during RL. |
| `max_prompt_length` | Maximum prompt token budget. |
| `max_completion_length` | Maximum completion token budget. |
| `max_eval_examples_per_subset` | Maximum checkpoint-eval examples per subset for that run. |

### Reward Configuration

| Column | Simple meaning |
| --- | --- |
| `Configured Initial Correctness Weight` | Starting correctness reward weight from the phase config. |
| `Configured Initial Formatting Weight` | Starting formatting reward weight from the phase config. |
| `Configured Initial Parseability Weight` | Starting parseability reward weight from the phase config. |
| `Configured Initial Finished Weight` | Starting finished-answer reward weight from the phase config. |
| `Configured Initial Tolerance Weight` | Starting tolerance reward weight from the phase config. |
| `Configured Initial Brevity Weight` | Starting brevity reward weight from the phase config. |
| `Correctness Weight` | Effective correctness reward weight saved with the checkpoint. |
| `Formatting Weight` | Effective formatting reward weight saved with the checkpoint. |
| `Parseability Weight` | Effective parseability reward weight saved with the checkpoint. |
| `Finished Weight` | Effective completion-finished reward weight saved with the checkpoint. |
| `Tolerance Weight` | Effective tolerance reward weight saved with the checkpoint. |
| `Brevity Weight` | Effective brevity reward weight saved with the checkpoint. |
| `Correctness Reward Mean` | Mean correctness reward observed during eval for this checkpoint. |
| `Format Reward Mean` | Mean formatting reward observed during eval for this checkpoint. |
| `Parseable Reward Mean` | Mean parseability reward observed during eval for this checkpoint. |
| `Finished Reward Mean` | Mean finished-answer reward observed during eval for this checkpoint. |
| `Tolerance Reward Mean` | Mean tolerance reward observed during eval for this checkpoint. |

### Evaluation Metrics

| Column | Simple meaning |
| --- | --- |
| `Exact Match` | Fraction of prompts whose final answer matched exactly. |
| `Tolerance Accuracy` | Fraction of prompts counted correct under numeric tolerance. |
| `Best-of-k Accuracy` | Best completion accuracy across the sampled completions per prompt. |
| `Best-of-k Tolerance Accuracy` | Best completion tolerance accuracy across sampled completions. |
| `Sample-Level Exact Match` | Exact-match rate measured at the sampled-completion level rather than prompt level. |
| `Sample-Level Tolerance Accuracy` | Tolerance accuracy measured at the sampled-completion level. |
| `Parseable Rate` | Fraction of outputs whose answer could be parsed successfully. |
| `Reasoning Tag Compliance` | Fraction of outputs that included the required reasoning tags correctly. |
| `Solution Tag Compliance` | Fraction of outputs that included the required solution tags correctly. |
| `Well-Formed Rate` | Positive-form version of malformed rate: higher means fewer malformed outputs. |
| `Malformed Rate` | Fraction of outputs with malformed structure or tags. |
| `Completion Success Rate` | Positive-form version of truncation rate: higher means fewer truncated outputs. |
| `Truncation Rate` | Fraction of outputs that ended before producing a usable final answer. |
| `Average Completion Tokens` | Mean output length in completion tokens. |
| `Repetition Rate` | Simple repetition score; higher means the model repeated itself more. |
| `Sample Diversity` | How varied the sampled answers were across multiple generations. |
| `Average Total Reward` | Mean total reward for the evaluated completions. |
| `Structure Score` | Composite structure-oriented checkpoint score used in selection. |
| `Correctness Score` | Composite correctness-oriented checkpoint score used in selection. |
| `Composite Score` | Main combined checkpoint score used for best-composite selection. |
| `KL Mean` | Mean KL divergence logged during training for that phase. |
| `KL P95` | 95th percentile KL divergence logged during training for that phase. |

### Checkpoint Selection Diagnostics

| Column | Simple meaning |
| --- | --- |
| `Checkpoint Rank Within Phase` | Checkpoint order within its phase after sorting by global step. |
| `Is Best Composite` | Whether the checkpoint is tagged as best composite for its phase. |
| `Is Best Correctness` | Whether the checkpoint is tagged as best correctness for its phase. |
| `Is Best Structure` | Whether the checkpoint is tagged as best structure for its phase. |
| `Is Latest` | Whether the checkpoint is tagged as the latest saved checkpoint for its phase. |
| `Controller History Length` | How many eval events were stored in the reward-controller history. |
| `Delta Exact vs Previous Milestone` | Change in exact match compared with the previous milestone row. |
| `Delta Composite vs Previous Milestone` | Change in composite score compared with the previous milestone row. |

### Evidence Paths

| Column | Simple meaning |
| --- | --- |
| `Checkpoint Path` | Relative checkpoint path recorded in the saved artifacts. |
| `Checkpoint Path Abs` | Absolute local filesystem path to the checkpoint directory or source file. |
| `Evidence Root` | Root folder containing the evidence for the row. |
| `Metrics Source Path` | File path that directly supplied the row metrics. |
| `Checkpoint Info Path` | Path to the saved checkpoint metadata JSON. |
| `Reward Weights Path` | Path to the saved effective reward weights JSON. |
| `Controller State Path` | Path to the saved reward-controller state JSON. |
| `Run Config Path` | Path to the saved run configuration JSON. |
| `Run Request Path` | Path to the saved CLI request JSON for the run. |
| `Diagnostics Path` | Path to the saved post-training diagnostics JSON for the phase. |
| `Train Log Summary Path` | Path to the saved training-log summary JSON for the phase. |
| `Summary Path` | Path to the short human-readable checkpoint summary file. |
| `Notes` | Extra caveats or context that did not fit cleanly into the other columns. |

## Analysis

- The baseline row starts at exact match 0.375, parseable rate 0.641, malformed rate 0.359, and truncation rate 0.242.
- Smoke Phase A moved exact match to 0.5 but still had weak structure, with parseable rate 0.5 and truncation rate 0.5.
- Smoke Phase B fixed structure on the tiny split: parseable rate reached 1, malformed and truncation both dropped to 0, but exact match stayed at 0.5.
- Smoke Phase C kept that strong structure profile but still sat at exact match 0.5, which shows the small split had likely hit a correctness ceiling.
- Smoke Phase D did not improve over Smoke Phase C; exact match stayed at 0.5 while structure regressed back to parseable rate 0.5 and truncation rate 0.5.
- Larger-split Phase C is the key jump: exact match rose to 0.75, composite score to 0.75, while structure stayed perfect.
- The same-notebook Phase D branch shows why checkpoint-aware selection matters: its latest checkpoint fell to exact match 0.5 even though the branch had reached 0.75 earlier.
- The dedicated Phase D rerun recovered to exact match 0.75, matching but not beating the recommended larger-split Phase C checkpoint.

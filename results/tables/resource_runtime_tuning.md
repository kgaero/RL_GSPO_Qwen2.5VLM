# Table C: Resource-Constrained Runtime Tuning

CSV source: `tables/resource_runtime_tuning.csv`

## Purpose

Configuration-and-outcome view showing how the stable Kaggle T4 profile differs from the reference default profile and what each successful run achieved under the constrained profile.

## Column Glossary

### Resource Runtime Tuning Columns

| Column | Simple meaning |
| --- | --- |
| `Run Family` | Run family or reference profile represented by the row. |
| `Notebook / Run Slug` | Notebook or run identifier tied to the row. |
| `Hardware` | Hardware used, or reference hardware label for comparison rows. |
| `Observed Constraint / Failure Mode` | Short statement of the memory or runtime constraint being addressed. |
| `Train Split` | Training split used by that run. |
| `Eval Split` | Evaluation split used by that run. |
| `Hardware Profile` | Named runtime profile applied to the run. |
| `Base Model` | Base pretrained model behind the run. |
| `4-bit Enabled` | Whether 4-bit loading was enabled. |
| `LoRA Rank` | LoRA rank used in that profile. |
| `Max LoRA Rank` | Maximum LoRA rank setting if present. |
| `max_seq_length` | Maximum sequence length allowed by the profile. |
| `image_size` | Image resolution used in the profile. |
| `num_generations` | Number of sampled completions per prompt during RL. |
| `max_prompt_length` | Maximum prompt token budget. |
| `max_completion_length` | Maximum completion token budget. |
| `gradient_accumulation_steps` | Gradient accumulation count used during training. |
| `gpu_memory_utilization` | Target vLLM/fast-generation memory reservation fraction. |
| `fast_inference enabled` | Whether the fast generation path remained enabled. |
| `vLLM version` | vLLM version if logged; otherwise not logged. |
| `cudagraph / compilation mode` | Compilation mode used to keep fast generation stable. |
| `Warm start used?` | Whether the run started from an earlier checkpoint rather than from scratch. |
| `What knob changed` | Short summary of the main profile differences versus the reference profile. |
| `Why this helps memory` | Plain-language reason the profile choices reduce memory pressure. |
| `Tradeoff introduced` | What capability or convenience was sacrificed to stay within the VRAM budget. |
| `Outcome` | What the row achieved under that profile. |

## Analysis

- The reference row shows the high-capacity default profile, while every successful Kaggle run used the much smaller `kaggle_t4` profile instead.
- The stable Kaggle T4 profile cut sequence length from 16384 to 1280, image size from 512 to 336, generations from 4 to 2, and LoRA rank from 16 to 8.
- The smoke Phase A row shows that this lower-memory profile was sufficient to validate the full RL loop on `testmini`.
- The large Phase C row shows the key resource-constrained result: the train split increased from `testmini` to `test` while the core T4 profile stayed the same, which supports the interpretation that split size mainly increased runtime rather than per-step VRAM.
- The same-notebook and dedicated Phase D rows show that the same T4-safe profile could support specialization runs as well, although the dedicated rerun was needed to preserve earlier outputs and recover the better final score.

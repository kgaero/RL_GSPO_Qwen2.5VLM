# Resource Knob Tradeoffs

CSV source: `tables/resource_knob_tradeoffs.csv`

## Purpose

Per-knob explanation of how the Kaggle T4 profile reduced memory pressure and what tradeoffs were accepted in exchange.

## Column Glossary

### Resource Knob Tradeoff Columns

| Column | Simple meaning |
| --- | --- |
| `Knob` | The runtime or training setting that changed. |
| `Default` | Reference value from the default higher-capacity profile. |
| `Kaggle T4` | Value used in the stable Kaggle T4 profile. |
| `Why it reduced memory pressure` | Simple reason this change lowered VRAM demand or stabilized runtime. |
| `Tradeoff accepted` | What was sacrificed in exchange for the lower memory demand. |

## Analysis

- The biggest headline reduction was sequence length: 16384 to 1280, which is about a 92.188% cut.
- Prompt budget dropped from 1024 to 320 and completion budget from 256 to 64, which traded long-form reasoning space for lower generation-time memory use.
- Image size fell from 512 to 336; in area terms that is roughly a 56.934% reduction in pixel workload for the vision stack.
- LoRA rank and number of generations were both halved, which directly reduced trainable state and multi-sample generation cost.
- The table makes the tradeoff explicit: almost every memory-saving change came with a loss in context budget, visual detail, or exploration diversity.

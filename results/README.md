# Results Package

This folder was generated from the current local and Kaggle-exported staged RL artifacts.

Recommended final checkpoint:
- `outputs_staged_large_continue/phase_c/checkpoint-120`
- exact match `0.75`
- tolerance accuracy `0.75`
- composite score `0.75`

Generated tables:
- `tables/master_table_milestones.csv`
- `tables/master_table_milestones.md`
- `tables/master_table_all_checkpoints.csv`
- `tables/master_table_all_checkpoints.md`
- `tables/resource_runtime_tuning.csv`
- `tables/resource_runtime_tuning.md`
- `tables/resource_knob_tradeoffs.csv`
- `tables/resource_knob_tradeoffs.md`
- `tables/runtime_stabilization_timeline.csv`
- `tables/runtime_stabilization_timeline.md`

Generated plots:
- `plots/evolution_panels.png`
- `plots/runtime_stabilization_timeline.png`
- `plots/checkpoint_frontier_scatter.png`
- `plots/checkpoint_heatmap.png`
- `plots/resume_lineage.png`

Regenerate with:
- `.venv-results/bin/python scripts/generate_results_report.py`

Evidence caveats:
- The larger-split Phase C best composite checkpoint remains the recommended final checkpoint.
- Dedicated Phase D matched the larger-split Phase C best score but did not exceed it.

Row counts: 27 checkpoint/audit rows, 12 milestone rows, 5 resource-tuning rows.

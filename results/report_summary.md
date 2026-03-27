# RL Fine-Tuning Result Summary

- Recommended final checkpoint: `outputs_staged_large_continue/phase_c/checkpoint-120`
- Recommended final metrics: exact `0.75`, tolerance `0.75`, parseable `1.0`, malformed `0.0`, truncation `0.0`
- Dedicated Phase D best matched but did not beat it: exact `0.75`, composite `0.75`
- Main causal read: staged RL fixed structure first; the larger split then produced the main correctness gain.
- Resource-constrained read: the larger split became feasible on Kaggle T4 because per-step memory pressure was reduced by the stable `kaggle_t4` profile, not because the problem became smaller.

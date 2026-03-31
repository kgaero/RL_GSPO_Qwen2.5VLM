# Controller Rule Audit

Source checkpoint table: `results/tables/master_table_all_checkpoints.csv`
Generated CSV: `/home/kgaer/code/RL_GSPO_Qwen2.5VLM/results/tables/controller_rule_audit.csv`

## Summary

- The controller audit covers 26 evaluated checkpoints across 7 phase resets.
- Parseability guard fired 2 times, format guard fired 2 times, and finish guard fired 2 times.
- Correctness escalation fired 12 times after structure was already stable.
- 6 correctness escalations happened on checkpoints where exact match had actually regressed, which makes the plateau rule's behavior explicit.
- The audit rows distinguish phase-default resets from controller-triggered updates, which the weight-evolution plot alone does not show.

## Compact Audit Table

| X Label | Phase Reset | Triggered Rules | Stable Structure | Correctness Plateau | Exact Previous | Exact Current | Exact Delta | Parseability Delta | Formatting Delta | Finished Delta | Correctness Delta | Controller Decision Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smk:a:49 | yes | parseable_guard;format_guard;finish_guard | no | no | None | 0.5 | None | 0.0 | 0.0 | 0.25 | 0.0 | yes |
| smk:b:60 | yes |  | yes | no | None | 0.5 | None | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| smk:b:120 | no | correctness_escalation | yes | yes | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| smk:b:180 | no | correctness_escalation | yes | yes | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| smk:b:240 | no | correctness_escalation | yes | yes | 0.5 | 0.0 | -0.5 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| smk:b:242 | no | correctness_escalation | yes | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| smk:c:60 | yes |  | yes | no | None | 0.0 | None | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| smk:c:119 | no |  | yes | no | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| smk:d:30 | yes | parseable_guard;format_guard;finish_guard | no | no | None | 0.5 | None | 0.25 | 0.25 | 0.25 | 0.0 | yes |
| lgc:c:60 | yes |  | yes | no | None | 0.25 | None | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| lgc:c:120 | no |  | yes | no | 0.25 | 0.75 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| lgc:c:180 | no | correctness_escalation | yes | yes | 0.75 | 0.5 | -0.25 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| lgc:c:240 | no | correctness_escalation | yes | yes | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| lgc:c:300 | no |  | yes | no | 0.5 | 0.75 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| lgc:c:360 | no | correctness_escalation | yes | yes | 0.75 | 0.5 | -0.25 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| lgc:c:420 | no | correctness_escalation | yes | yes | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| lgc:c:480 | no |  | yes | no | 0.5 | 0.75 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| lgc:c:540 | no | correctness_escalation | yes | yes | 0.75 | 0.5 | -0.25 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| lgc:c:600 | no | correctness_escalation | yes | yes | 0.5 | 0.25 | -0.25 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| lgc:c:602 | no |  | yes | no | 0.25 | 0.75 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| lgc:d:60 | yes |  | yes | no | None | 0.75 | None | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| lgc:d:120 | no | correctness_escalation | yes | yes | 0.75 | 0.25 | -0.5 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| lgc:d:130 | no |  | yes | no | 0.25 | 0.5 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| dpd:d:60 | yes |  | yes | no | None | 0.25 | None | 0.0 | 0.0 | 0.0 | 0.0 | yes |
| dpd:d:120 | no | correctness_escalation | yes | yes | 0.25 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | yes |
| dpd:d:130 | no |  | yes | no | 0.25 | 0.75 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | yes |

## Column Notes

- `X Label`: Short checkpoint label used in plots.
- `Phase Reset`: Whether this row is the first controller update in its phase run.
- `Triggered Rules`: Semicolon-separated list of controller rules that fired on this checkpoint.
- `Stable Structure`: Whether all structure stability thresholds were satisfied.
- `Correctness Plateau`: Whether exact-match gain stayed below the plateau threshold over the stable window.
- `Exact Previous`: Previous exact-match value used by the plateau rule.
- `Exact Current`: Current exact-match value seen by the controller.
- `Exact Delta`: Current exact minus previous exact used by the plateau rule.
- `Parseability Delta`: Change applied to parseability weight on this checkpoint.
- `Formatting Delta`: Change applied to formatting weight on this checkpoint.
- `Finished Delta`: Change applied to finished-answer weight on this checkpoint.
- `Correctness Delta`: Change applied to correctness weight on this checkpoint.
- `Controller Decision Match`: Whether the reconstructed post-update weights matched the saved checkpoint weights.

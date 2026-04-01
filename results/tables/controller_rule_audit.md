# Controller Rule Audit

CSV source: `tables/controller_rule_audit.csv`

## Purpose

Checkpoint-by-checkpoint audit of which reward-controller rules fired, what evidence triggered them, and which reward weights changed.

## Column Glossary

### Controller Audit Columns

| Column | Simple meaning |
| --- | --- |
| `Timeline Order` | Chronological order used in the main checkpoint plots. |
| `X Label` | Short checkpoint label used in plots. |
| `Run Family` | Internal run family for the checkpoint. |
| `Run Family Label` | Human-friendly run family label. |
| `Phase` | Training phase for the checkpoint. |
| `Phase Label` | Human-friendly phase label. |
| `Checkpoint` | Checkpoint directory name. |
| `Global Step` | Trainer global step at save time. |
| `Phase Reset` | Whether this row is the first controller update in its phase run. |
| `Parseable Guard Fired` | Whether the low-parseability guard condition fired. |
| `Format Guard Fired` | Whether the tag/malformed formatting guard condition fired. |
| `Finish Guard Fired` | Whether the truncation or overlength guard condition fired. |
| `Correctness Rule Fired` | Whether correctness weight escalation fired after a stable plateau. |
| `Stable Structure` | Whether all structure stability thresholds were satisfied. |
| `Stable Window Ready` | Whether enough checkpoint history existed to test the correctness plateau rule. |
| `Correctness Plateau` | Whether exact-match gain stayed below the plateau threshold over the stable window. |
| `Triggered Rules` | Semicolon-separated list of controller rules that fired on this checkpoint. |
| `Changed Components` | Semicolon-separated list of reward weights that actually changed. |
| `Clamped Components` | Semicolon-separated list of weights clipped by min/max bounds. |
| `Exact Previous` | Previous exact-match value used by the plateau rule. |
| `Exact Current` | Current exact-match value seen by the controller. |
| `Exact Delta` | Current exact minus previous exact used by the plateau rule. |
| `Parseable Rate` | Current parseable-answer rate seen by the controller. |
| `Solution Tag Compliance` | Current solution-tag compliance seen by the controller. |
| `Reasoning Tag Compliance` | Current reasoning-tag compliance seen by the controller. |
| `Malformed Rate` | Current malformed-answer rate seen by the controller. |
| `Truncation Rate` | Current truncation rate seen by the controller. |
| `Average Completion Tokens` | Current average completion length seen by the controller. |
| `Average Token Fraction` | Average completion tokens divided by max completion length. |
| `Max Completion Length` | Max completion length used for the controller decision. |
| `Correctness Before` | Correctness weight before the controller update. |
| `Correctness After` | Correctness weight after the controller update. |
| `Correctness Delta` | Change applied to correctness weight on this checkpoint. |
| `Formatting Before` | Formatting weight before the controller update. |
| `Formatting After` | Formatting weight after the controller update. |
| `Formatting Delta` | Change applied to formatting weight on this checkpoint. |
| `Parseability Before` | Parseability weight before the controller update. |
| `Parseability After` | Parseability weight after the controller update. |
| `Parseability Delta` | Change applied to parseability weight on this checkpoint. |
| `Finished Before` | Finished-answer weight before the controller update. |
| `Finished After` | Finished-answer weight after the controller update. |
| `Finished Delta` | Change applied to finished-answer weight on this checkpoint. |
| `Controller Decision Path` | Path to the saved per-checkpoint controller decision artifact when present. |
| `Controller Decision Source` | Whether the audit row came from a saved artifact or report-side reconstruction. |
| `Controller Decision Match` | Whether the reconstructed post-update weights matched the saved checkpoint weights. |

## Analysis

- The controller audit covers 26 evaluated checkpoints across 7 phase resets.
- Parseability guard fired 2 times, format guard fired 2 times, and finish guard fired 2 times.
- Correctness escalation fired 12 times after structure was already stable.
- 6 correctness escalations happened on checkpoints where exact match had actually regressed, which makes the plateau rule's behavior explicit.
- The audit rows distinguish phase-default resets from controller-triggered updates, which the weight-evolution plot alone does not show.

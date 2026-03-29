# Runtime Stabilization Timeline

CSV source: `tables/runtime_stabilization_timeline.csv`

## Purpose

Chronological milestone view showing how the project moved from the weak baseline to smoke validation, larger-split correctness gains, and dedicated Phase D specialization.

## Column Glossary

### Runtime Stabilization Timeline Columns

| Column | Simple meaning |
| --- | --- |
| `Timeline Order` | Chronological order used in the timeline figure and table. |
| `Event Label` | Short label for the milestone event. |
| `Milestone ID` | Stable internal milestone key. |
| `Run Family` | Run family that produced the milestone. |
| `Phase` | Training phase tied to the milestone. |
| `Train Split` | Training split used at that milestone. |
| `Eval Split` | Eval split used at that milestone. |
| `Warm Start` | Resume alias or explicit warm-start checkpoint used before training. |
| `Status` | Short status label describing what changed at that milestone. |
| `Best Exact` | Best exact-match value represented by the milestone. |
| `Best Composite` | Best composite score represented by the milestone. |
| `Key Change` | Short description of what the training setup changed at that point. |
| `Interpretation` | Plain-language read of why the milestone matters. |

## Analysis

- The timeline contains 8 milestone events: Baseline -> Smoke Phase A -> Smoke Phase B -> Smoke Phase C -> Smoke Phase D -> Large Phase C -> Large Phase D -> Dedicated Phase D.
- It shows a clear order of improvement: baseline weakness, smoke-run validation, smoke-run structure stabilization, then larger-split correctness improvement.
- The smoke Phase D milestone marks the point where the small split stopped helping, which is why the next successful move was to scale the data rather than keep iterating on the same tiny split.
- Large Phase C is the first milestone where correctness clearly improved beyond the smoke plateau, and both later Phase D branches should be read as specialization branches rather than primary replacements for that checkpoint.

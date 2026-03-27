# RL Fine-Tuning Results PRD

## Purpose

This document captures the agreed reporting scope for the Qwen2.5-VL staged RL fine-tuning work. It is a living agreement file and should be updated as reporting requirements evolve.

Primary goal:
- produce a clear, evidence-backed report showing what the RL system achieved, why it worked, and how metrics evolved over time

Secondary goal:
- preserve enough detail that the report is useful for learning and post-hoc debugging, not only for executive summary claims

## Reporting Principles

- Keep both a milestone-level view and a full checkpoint-level audit trail.
- Prefer evidence over narrative. Every major claim should be traceable to saved metrics, aliases, or checkpoint artifacts.
- Separate structure improvement from correctness improvement.
- Do not assume latest checkpoint is best. Always show alias-selected best checkpoints.
- Make the figure easy to read top-to-bottom with a shared x-axis.
- For structure plots, use positive-form metrics where possible so improvements move upward.
- Do not invent or imply measurements that were not actually logged. If a quantity was not measured, explain it with configuration-and-outcome evidence instead of fabricating a time-series plot.
- If a planned milestone artifact is unavailable in the currently exported bundles, keep the milestone row with explicit `artifact_missing` status rather than backfilling unsupported values.

## Agreed Deliverables

### 1. Master Table A: Milestones Only

Purpose:
- compact report-ready summary
- compare major runs, best checkpoints, important regressions, and final selections

Row scope:
- baseline pre-refactor eval snapshot
- smoke Phase A best
- smoke Phase B best
- smoke Phase C best
- smoke Phase D best
- larger-split Phase C best
- larger-split Phase C latest
- dedicated Phase D best
- any explicitly important regression checkpoints

### 2. Master Table B: Every Checkpoint Row

Purpose:
- detailed audit table for learning and debugging
- support checkpoint-selection analysis
- support appendix/report supplement

Row scope:
- every published checkpoint with evaluation metrics
- include alias membership and keep/discard decisions

### 3. Figure A: Vertically Stacked Multi-Panel Evolution Figure

Layout:
- one figure
- multiple panels stacked vertically
- shared x-axis
- phase boundaries clearly marked
- easier to read top-to-bottom and compare changes over the same checkpoints

Positive-form structure metrics:
- `well_formed_rate = 1 - malformed_answer_rate`
- `completion_success_rate = 1 - truncation_rate`

### 4. Table C: Resource-Constrained Runtime Tuning Table

Purpose:
- explain how the RL system was made to run on Kaggle T4
- show what resource problem existed
- show which knobs were changed
- show what tradeoffs were accepted to stay within memory limits

This table is configuration-and-outcome based. It should not imply unavailable GPU-memory telemetry.

### 5. Figure / Diagram B: Failure-to-Stable Runtime Timeline

Purpose:
- show the engineering progression from failed Kaggle runs to stable smoke-test and larger-split runs
- explain that success came from iterative runtime stabilization, not from a single lucky configuration

## Recommended X-Axis Convention

Primary x-axis:
- checkpoint progression within chronological run order

Recommended labels:
- `baseline`
- `phase_a:49`
- `phase_b:60`
- `phase_b:120`
- ...
- `large_phase_c:120`
- ...
- `phase_d_only:120`
- `phase_d_only:130`

Visual aids:
- vertical separators between phases
- optional background shading per phase family
- star marker for `best_composite`
- diamond marker for `best_correctness`
- square marker for `best_structure`
- hollow circle for `latest`

## Table Schemas

### Table A: Milestones Only

Use the widest useful schema so the same table can support both reporting and analysis.

Columns:
- `Run Family`
- `Notebook / Run Slug`
- `Output Root`
- `Phase`
- `Checkpoint`
- `Global Step`
- `Alias Role`
- `Keep / Discard`
- `Decision Reason`
- `Key Interpretation`
- `Train Split`
- `Eval Split`
- `Hardware Profile`
- `Seed / Resume Source`
- `Warm-Start Checkpoint`
- `Training Strategy Introduced`
- `Stage Mix / Curriculum`
- `Base Model`
- `Correctness Weight`
- `Formatting Weight`
- `Parseability Weight`
- `Finished Weight`
- `Tolerance Weight`
- `Brevity Weight`
- `Correctness Reward Mean`
- `Format Reward Mean`
- `Parseable Reward Mean`
- `Finished Reward Mean`
- `Tolerance Reward Mean`
- `Exact Match`
- `Tolerance Accuracy`
- `Best-of-k Accuracy`
- `Best-of-k Tolerance Accuracy`
- `Sample-Level Exact Match`
- `Sample-Level Tolerance Accuracy`
- `Parseable Rate`
- `Reasoning Tag Compliance`
- `Solution Tag Compliance`
- `Well-Formed Rate`
- `Malformed Rate`
- `Completion Success Rate`
- `Truncation Rate`
- `Average Completion Tokens`
- `Repetition Rate`
- `Sample Diversity`
- `Average Total Reward`
- `Structure Score`
- `Correctness Score`
- `Composite Score`
- `Delta Exact vs Previous Milestone`
- `Delta Composite vs Previous Milestone`

### Table B: Every Checkpoint Row

Use all columns from Table A plus:
- `Checkpoint Path`
- `Checkpoint Rank Within Phase`
- `Is Best Composite`
- `Is Best Correctness`
- `Is Best Structure`
- `Is Latest`
- `Controller History Length`
- `Reward Weights Source`
- `Notes`

### Table C: Resource-Constrained Runtime Tuning Table

Columns:
- `Run Family`
- `Notebook / Run Slug`
- `Hardware`
- `Observed Constraint / Failure Mode`
- `Train Split`
- `Eval Split`
- `Hardware Profile`
- `Base Model`
- `4-bit Enabled`
- `LoRA Rank`
- `Max LoRA Rank`
- `max_seq_length`
- `image_size`
- `num_generations`
- `max_prompt_length`
- `max_completion_length`
- `gradient_accumulation_steps`
- `gpu_memory_utilization`
- `fast_inference enabled`
- `vLLM version`
- `cudagraph / compilation mode`
- `Warm start used?`
- `What knob changed`
- `Why this helps memory`
- `Tradeoff introduced`
- `Outcome`

Recommended values for `Tradeoff introduced`:
- longer runtime
- less generation diversity
- shorter reasoning budget
- lower visual resolution
- lower sequence capacity
- smaller eval coverage per checkpoint

Purpose of this table:
- make the resource problem concrete
- show the memory-aware decisions explicitly
- make the engineering tradeoff legible
- compare the stable `kaggle_t4` profile against the higher-capacity default reference profile when that contrast is more informative than comparing successful runs against each other

## Figure Specification

### Panel 1: Reward Weight Evolution

Lines:
- correctness weight
- formatting weight
- parseability weight
- finished weight
- tolerance weight
- brevity weight

Purpose:
- show how metric-gated control shifted training emphasis

Rendering:
- step lines preferred over smoothed curves
- markers at actual checkpoint evaluations

### Panel 2: Structure Evolution

Lines:
- parseable rate
- reasoning tag compliance
- solution tag compliance
- well-formed rate
- completion success rate

Purpose:
- show when response format stabilized

### Panel 3: Accuracy Evolution

Lines:
- exact match
- tolerance accuracy
- best-of-k accuracy
- composite score

Purpose:
- show when correctness improved and whether gains held

### Panel 4: Generation Behavior

Lines:
- average completion tokens
- repetition rate
- sample diversity
- average total reward

Purpose:
- show whether shorter / cleaner completions correlated with better outcomes

### Panel 5: Optional Decision Overlay

Options:
- keep/discard markers on top of checkpoints
- or small strip/heat row under the x-axis

Purpose:
- show chosen checkpoints without mixing categorical values into numeric panels

Decision:
- include this as a fifth thin panel in the main stacked figure
- encode alias role (`best_composite`, `best_correctness`, `best_structure`, `latest`) and `keep/discard` status

## Confirmed High-Level Findings To Support In Report

### Finding 1

The staged RL system made structure reliable.

Evidence pattern:
- parseable rate increased
- malformed rate dropped
- truncation dropped
- completion length dropped sharply

### Finding 2

Larger-split continuation produced the main correctness improvement.

Evidence pattern:
- smoke runs mainly stabilized structure
- larger-split Phase C improved exact/tolerance accuracy materially

### Finding 3

Checkpoint selection mattered because training was non-monotonic.

Evidence pattern:
- exact/composite scores oscillated across checkpoints
- best alias often differed from intuitive later checkpoints

### Finding 4

Phase D specialized training did not clearly surpass the best larger-split Phase C checkpoint.

Evidence pattern:
- Phase D recovered to a strong result
- but did not exceed the best larger-split Phase C score

## Current Narrative Framing

Preferred summary:
- the new RL training schedule primarily fixed structure and made learning stable
- the larger split primarily drove the correctness gain
- together they enabled successful RL fine-tuning on constrained Kaggle T4 hardware

Short causal summary:
- structure gains: mostly from staged RL design
- correctness gains: mostly from larger split, enabled by staged RL design

Checkpoint recommendation:
- recommend the larger-split Phase C best composite checkpoint as the primary final checkpoint
- present Phase D as a specialization branch that recovered to the same quality band but did not exceed the best larger-split Phase C result

## Resource-Constrained Training Story

This report should explicitly document that the RL pipeline was made practical on Kaggle T4 hardware by reducing per-step memory pressure while preserving the staged RL design.

### Core explanation

Key point:
- larger dataset size mostly increases runtime, not per-step VRAM
- per-step VRAM was controlled by runtime/profile choices, not by the train split size itself

Primary memory-saving interventions:
- 4-bit Unsloth LoRA training instead of full fine-tuning
- low-rank LoRA profile
- reduced `max_seq_length`
- reduced `image_size`
- reduced `max_prompt_length`
- reduced `max_completion_length`
- constrained `num_generations`
- conservative `gpu_memory_utilization`
- vLLM kept enabled but tuned for T4-safe piecewise cudagraph execution
- tiny checkpoint-time evaluation on `testmini`

### Resource-constrained achievements to visualize

- the RL system ran end-to-end on Kaggle T4 despite repeated earlier failures
- model load, vLLM init, and generation path were stabilized under low VRAM
- larger-split continuation became feasible without changing the core RL design
- resume / continuation from selected checkpoints remained usable on the constrained profile

### Recommended resource-constrained visuals

#### A. Runtime tuning table

Columns:
- `Profile`
- `max_seq_length`
- `image_size`
- `lora_rank`
- `num_generations`
- `max_prompt_length`
- `max_completion_length`
- `gpu_memory_utilization`
- `fast_inference mode`
- `compilation mode`
- `Outcome`

Rows:
- original / pre-stable setup
- intermediate failed Kaggle profiles if you want the engineering story
- final `kaggle_t4` stable profile

Purpose:
- makes it obvious which knobs were changed to fit T4

#### B. Failure-to-fix timeline

Rows or bars:
- P100 unsupported
- vLLM cache failure
- duplicate layer / cache block issue
- dtype mismatch
- image payload failure
- zero trainable params
- OOM at first GRPO step
- successful T4 smoke run
- successful larger-split continuation

Purpose:
- shows that the resource-constrained success was engineered, not accidental

#### C. Memory-sensitive settings panel

A compact annotated panel listing:
- `seq length`
- `image size`
- `completion length`
- `num_generations`
- `LoRA rank`

Purpose:
- supports the claim that the final working setup came from aggressive memory-aware tuning

#### D. Runtime outcome matrix

Columns:
- `Notebook`
- `Split`
- `Hardware`
- `Profile`
- `Warm start`
- `Completed?`
- `Best exact`
- `Best composite`

Purpose:
- links hardware feasibility to training outcomes

#### E. Resource tradeoff summary panel

This can be a compact textual or visual summary that explicitly pairs each memory-saving change with its downside.

Suggested 3-column format:
- `Knob changed`
- `Why it reduced memory pressure`
- `Tradeoff accepted`

Example pairs:
- shorter sequence length -> lower KV/cache and activation footprint -> less room for long-context reasoning
- smaller image size -> cheaper vision forward pass -> lower visual detail fidelity
- fewer generations -> less generation-time memory -> weaker exploration / diversity
- shorter completion length -> lower generation memory and lower truncation risk -> less room for verbose chains of thought
- lower LoRA rank -> lighter trainable state -> lower adaptation capacity
- smaller eval subset -> lighter checkpoint evaluation -> noisier checkpoint selection

### Recommended explanatory wording

Use language like:
- “We made larger-split RL training feasible on Kaggle T4 by reducing per-step memory demand rather than by simplifying the training objective.”
- “The critical shift was a memory-aware runtime profile: shorter sequences, smaller image resolution, fewer generations, low-rank LoRA, and T4-safe vLLM compilation.”
- “This allowed the same staged RL system to scale from smoke-test `testmini` runs to a larger split under the same VRAM budget.”

## Additional Visualization / Analysis Ideas

These are optional but valuable, especially for a first complex RL project.

### A. Checkpoint Selection Scatter

Plot:
- x = structure score
- y = correctness score
- point size = composite score
- color = phase

Why:
- reveals whether checkpoints lie on a structure/correctness tradeoff frontier

### B. Failure-Mode Stacked Bars

Buckets:
- correct
- parseable but wrong
- malformed
- truncated

Why:
- explains what type of failure dominated each stage

### C. Heatmap of Checkpoints x Metrics

Rows:
- checkpoints

Columns:
- exact, tolerance, parseable, well-formed, completion success, tokens, composite, reward weights

Why:
- useful dense appendix figure

### D. Delta Waterfall

Steps:
- baseline
- smoke structure stabilization
- larger split continuation
- Phase D specialization

Why:
- shows which intervention contributed most

### E. Reward-Weight vs Metric Correlation View

Plot:
- correctness weight against exact match
- formatting/parseability weights against structure metrics

Why:
- helps explain why metric-gated reward shaping worked

### F. Length vs Correctness Scatter

Plot:
- x = average completion tokens
- y = exact match

Why:
- supports the argument that shorter, non-truncated outputs were helpful

### G. Resume / Lineage Diagram

Diagram:
- smoke Phase C best -> larger-split Phase C -> dedicated Phase D

Why:
- clearly explains continuation workflow and checkpoint reuse

## Suggested Report Structure

1. Problem statement
- original RL setup suffered from malformed outputs, truncation, weak parseability, and weak policy movement

2. Intervention
- staged curriculum
- modular rewards
- metric-gated reward weighting
- checkpoint-aware checkpoint selection
- practical resume / continuation workflow

3. Results
- structure stabilized first
- correctness improved after larger-split continuation
- checkpoint-aware selection was necessary
- Phase D matched but did not beat the best Phase C checkpoint

4. Resource-Constrained Feasibility
- explain the memory problem using configuration-and-outcome table data
- show which knobs were changed to fit Kaggle T4
- show what tradeoff was sacrificed for the gain, especially runtime and reduced capacity budget

5. Conclusion
- best current deployable checkpoint is the larger-split Phase C best composite checkpoint

## Open / To Be Decided

- whether the final report body should show only milestone rows or both milestone and every-checkpoint tables inline
- whether to include reward-component means in the main table or appendix table only
- whether to add a checkpoint frontier scatter to the main report

## Working Rule For Future Updates

When new reporting decisions are made, update this file first so:
- table schemas stay stable
- figure design remains consistent
- later code generation matches the agreed report perimeter

## Current Package List

Main report package:
- Table A: milestone-only master table
- Figure A: stacked multi-panel evolution figure with shared x-axis and a thin keep/discard panel
- Table C: resource-constrained runtime tuning table
- Diagram B: failure-to-stable runtime timeline

Appendix package:
- Table B: every-checkpoint table
- checkpoint frontier scatter
- checkpoint heatmap
- lineage / resume diagram

Resource-constrained explanation package:
- explain the problem with configuration-and-outcome table data
- explicitly list what knobs changed to address the problem
- explicitly list the tradeoff sacrificed for the gain, especially runtime and reduced capacity / diversity / context budget

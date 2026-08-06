# Fluid-Structure-Growth Weak Coupling Against the Real Load Trajectory

## Problem Statement

svFSGe partitions a fluid-structure-growth (FSG) aneurysm simulation into
alternating fluid (blood flow) and solid (vessel wall growth & remodeling,
"G&R") solves, coupled at the fluid-solid interface. The solid side is
driven by a spatial "insult" -- a localized elastin knock-down that models
aneurysm initiation -- ramped in over pseudo-time by the solver itself,
following the real 80-step production load curve
(`fsg.p["gr_load"]`, `f(t) = tanh(2·t/80)/tanh(2)`, `t=0..80`). `nloads=80`
for this experiment, so a fully successful run traverses the whole real
trajectory and reaches true full load (`f(80)=1.0`) at the end.

`run_weak_coupling(fsg, t_start, i_start)` is the per-load-step control
loop. Its job is orchestrating `fsg.step("fluid"/"mesh"/"solid", ...)` and
whatever interface-displacement relaxation/acceleration scheme it uses
between steps -- **not** driving the insult level, which the solid solver
already applies internally from `fsg.p["gr_load"]`'s curve. It must track
how far through this run's own step budget it got (as a scalar `sₜ ∈
[0,1]`, `t/nloads`) for scoring, and may optionally override the insult
level deliberately (e.g. to retry a failed step at a reduced level via
`fsg.set_gr_insult()`) -- but doing so on every step by default would
double-ramp on top of the curve's own values and produce a different,
wrong loading history (this is exactly the bug an earlier version of this
seed had, and exactly the bug a real commit to this repo fixed in the
CI's own weak-coupling test).

**Input:** a live `FSG` instance (`fsg`) already configured with the load
curve (`fsg.p["gr_load"]`), the spatial insult profile it scales
(`fsg.p["gr_insult"]`), and `fsg.p["nloads"]` load steps for this run. The
mesh, solver invocation (`fsg.step`), and all file I/O are fixed
infrastructure -- do not reimplement or bypass them, and never touch the
C++ solver internals.

**Output:** none returned directly. The function's effect is (a) advancing
`fsg`'s internal state via `fsg.step(...)` calls and (b) writing
`weak_omega_history.json` incrementally (one entry per successfully
completed load step, each including the progress scalar `"s"` for that
step) -- this file is how scoring recovers progress even if the process is
killed by the timeout mid-run.

## Formal Specification

**Variables:**

- Load step index t = 0, ..., nloads (t=0 is prestress, no G&R load yet)
- Progress scalar sₜ = t / nloads ∈ [0, 1] -- fraction of THIS run's step
  budget reached, not the physical load fraction (see gr_load above)
- Interface displacement history used for any relaxation/acceleration
  scheme you choose to keep or replace

**Objective:** reach sₜ = 1.0 (complete all 80 steps of the real
trajectory without failing, i.e. true full load), while keeping the final
displacement field close to the true converged solution at full load, and
minimizing the number of fluid solves performed.

**Constraints:**

1. Every load step must advance strictly through `fsg.step(...)` -- no
   direct solver invocation, no skipping steps.
2. On any solver failure, call `fsg._save_failure_case(t, i)` and return.
3. `weak_omega_history.json` must be written (via `_save_weak_history()`
   or equivalent) after every successfully completed step, not only at
   the end.
4. Do not rewrite `fsg.p["gr_load"]` -- it's the real trajectory, not a
   parameter to tune.

## Evaluation

Score = 0.4 × stability + 0.4 × accuracy + 0.2 × efficiency, each in
[0, 1].

- **Stability (0.4):** the last progress scalar successfully reached
  (`s_final`, from the last entry of `weak_omega_history.json`), clamped
  to [0, 1]. 1.0 means the run completed all nloads steps without failing;
  a lower value is the fraction of this run's step budget actually
  completed before it stopped.

- **Accuracy (0.4):** compares the run's last successfully converged
  state (`partitioned/converged/tube_*.vtu`, highest step number) against
  a tight-tolerance ground-truth reference that reaches the SAME final
  target (`test_reference/vanilla_tolfinal_n10`, strong coupling, full
  static `gr_insult.mag` applied via a faster 10-step ramp instead of the
  real 80-step one) -- valid because what's compared is the converged
  endpoint, not the pacing used to reach it. Displacement field only,
  computed the same way the CI's own `scripts/compare_results.py` checks
  displacement (`atol = rtol = 1e-8`, matching svMultiPhysics' own test
  tolerance):
  `err = mean(max(0, (|diff| - atol - rtol·|ref|) / (atol + rtol·|ref|)))`,
  then `score = exp(-k·err)` with `k` calibrated so a uniform 5% relative
  error scores 0.65. This is computed unconditionally -- even a run that
  didn't complete all 80 steps (`s_final < 1`) is compared against the
  full-load reference as-is, so it will naturally score low rather than
  being hard-zeroed. (Confirmed empirically: the seed's true full-load
  state gives ~5% median relative displacement error against this
  reference -- comparing mismatched load levels instead, e.g. an earlier,
  buggy version of this scoring, gave ~65% error on the exact same run.)

- **Efficiency (0.2):** 0 unless `s_final == 1.0` exactly (no credit for
  cheap-but-incomplete runs). Otherwise, based on the number of fluid
  solves performed (`cost`, counted from `partitioned/fluid_*.log`):
  `score = min(1 - log(cost/c_min) / log(c_max/c_min), 1)` with
  `c_min = 3`, `c_max = 200`.

**Timeout:** 30 minutes per evaluation.

## Solution Guidance

**Known approaches:**

- **The seed:** pure coupling logic -- one fluid + one solid solve per
  load step, inter-step Aitken Δ² relaxation of the interface displacement
  (persisting `omega` across load steps), no insult-level control at all
  (the built-in `gr_load` curve handles that). `s_t = t/nloads` is tracked
  purely for reporting, not used to drive anything.
- **Residual-adaptive relaxation:** grow/shrink `omega`'s bounds or use a
  different acceleration scheme (IQN-ILS-style history, better
  predictors) based on how large the residual (`mean_r` in the history)
  has been -- may let the run stay stable with fewer wasted fluid solves,
  or reach a step it would otherwise fail at.
- **Backoff-and-retry:** on a solid-solve failure, instead of giving up
  immediately, deliberately retry the same step at a reduced insult level
  (via `fsg.set_gr_insult()`, interpolating toward the curve's target for
  that step) before advancing -- trades extra fluid solves (hurts
  efficiency) for a shot at still reaching `s_final == 1.0` (worth much
  more, since it gates both stability and efficiency). Be deliberate about
  this, not routine -- see the double-ramping note in `initial_program.py`.

**What makes a good solution:**

- Reaches `s_final == 1.0` reliably (this gates both stability and
  efficiency credit) using as few fluid solves as possible.
- Stays numerically close to the true converged displacement field --
  large deviations (even at `s_final == 1.0`) mean the coupling scheme is
  cutting corners on accuracy to get there.

**Common pitfalls:**

- Reaching `s_final == 1.0` via a route that produces a badly diverged or
  unrealistic displacement field (e.g. skipping intermediate steps) will
  score well on stability/efficiency but badly on accuracy.
- Forgetting to write `weak_omega_history.json` after *every* successful
  step (not just the last) makes the run unscoreable if it's killed by the
  timeout partway through.
- Rewriting `fsg.p["gr_insult"]["mag"]` on every step by default (rather
  than only as a deliberate, occasional override) double-ramps against
  the solver's own internal application of `fsg.p["gr_load"]`'s curve,
  producing a different (and worse) loading history than intended. This
  is a real bug that shipped in an earlier version of this seed.

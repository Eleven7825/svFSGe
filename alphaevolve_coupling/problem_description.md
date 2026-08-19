# Fluid-Structure-Growth Coupling via Two Black-Box Primitives

## Problem Statement

svFSGe partitions a fluid-structure-growth (FSG) aneurysm simulation into
alternating fluid (blood flow, "F") and solid (vessel wall growth &
remodeling, "S") solves, coupled at the fluid-solid interface. The solid
side grows toward a target pathology (a localized elastin knock-down, "the
insult") as a ramp fraction `s` goes from 0 (ungrown) to 1 (fully grown,
the real production target) is applied.

`run_coupling(fsg, t_start, i_start)` builds its own schedule by calling
two black-box primitives, `F()` and `S(s)`, in whatever pattern, order, or
frequency it decides:

  - `F()` -- one fluid(+mesh) solve on the current wall.
  - `S(s)` -- one solid solve, applying ramp fraction `s` directly (`s`
    drives the material model's internal blending of every growth-related
    property, not just the insult magnitude -- it's the single lever for
    "how grown is the wall at this step").

Neither takes any other argument; all the bookkeeping (call counters,
mesh-warp timing, load-curve extension) is handled for you. What's being
optimized is **the arrangement of F/S calls and the `s` schedule** that
gets `S` to `s=1.0` (true full load) as cheaply and accurately as
possible -- not any particular fixed number of steps or fixed pacing.

**Input:** a live `FSG` instance (`fsg`) already configured with the
target insult profile (`fsg.p["gr_insult"]`, fixed -- never rewrite this)
and a generous step-count safety cap (`fsg.p["nloads"]`, currently 80 --
you don't have to use all of it, and going past it just clamps rather than
crashing). The mesh, solver invocation, and all file I/O are fixed
infrastructure, already wrapped by F/S -- never call `fsg.step(...)`
directly, and never touch the C++ solver internals.

**Output:** none returned directly. The function's effect is (a)
advancing `fsg`'s internal state via `F()`/`S(s)` calls and (b) writing
`weak_omega_history.json` incrementally (one entry per successful `S(s)`
call, including the `s` used) -- this file is how scoring recovers
progress even if the process is killed by the timeout mid-run.

## Formal Specification

**Variables:**

- Ramp fraction s ∈ [0, 1], chosen freely at each `S(s)` call (0 = ungrown
  prestress, 1 = true full load -- the real production target)
- Call pattern: any sequence of `F()`/`S(s)` calls, in any order, any
  count
- Interface displacement history used for any relaxation/acceleration
  scheme you choose to keep or replace

**Objective:** reach s = 1.0 without failing, while keeping the final
displacement field close to the true converged solution at full load, and
minimizing the number of `F()` calls performed.

**Constraints:**

1. Every solve must go through `F()`/`S(s)` -- no direct solver
   invocation.
2. On any solver failure (`F()`/`S(s)` returning `True`), call
   `fsg._save_failure_case(t, i)` and return.
3. `weak_omega_history.json` must be written after every successful
   `S(s)` call, not only at the end.
4. Never rewrite `fsg.p["gr_insult"]` -- the target pathology is fixed;
   `s` is the only thing you control.

## Evaluation

Score = 0.4 × stability + 0.4 × accuracy + 0.2 × efficiency, each in
[0, 1].

- **Stability (0.4):** the last `s` successfully reached (`s_final`, from
  the last entry of `weak_omega_history.json`), clamped to [0, 1]. 1.0
  means the run reached true full load without failing; a lower value is
  how far it got before something crashed.

- **Accuracy (0.4):** compares the run's last successfully converged
  state (`partitioned/converged/tube_*.vtu`, highest step number) against
  a tight-tolerance ground-truth reference that reaches the SAME final
  target (`test_reference/vanilla_tolfinal_n10`, strong coupling, full
  `gr_insult` applied via a faster 10-step ramp instead of a slower one)
  -- valid because what's compared is the converged endpoint, not the
  pacing used to reach it. Displacement field only, computed the same way
  the CI's own `scripts/compare_results.py` checks displacement
  (`atol = rtol = 1e-8`, matching svMultiPhysics' own test tolerance):
  `err = mean(max(0, (|diff| - atol - rtol·|ref|) / (atol + rtol·|ref|)))`,
  then `score = exp(-k·err)` with `k` calibrated so a uniform 5% relative
  error scores 0.65. Computed unconditionally -- even a run that didn't
  reach s=1.0 is compared against the full-load reference as-is, so it
  will naturally score low rather than being hard-zeroed.

- **Efficiency (0.2):** 0 unless `s_final == 1.0` exactly (no credit for
  cheap-but-incomplete runs). Otherwise, based on the number of `F()`
  calls performed (`cost`, counted from `partitioned/fluid_*.log`):
  `score = min(1 - log(cost/c_min) / log(c_max/c_min), 1)` with
  `c_min = 3`, `c_max = 200`.

**Timeout:** 30 minutes per evaluation.

## Solution Guidance

**Known approaches:**

- **The seed:** one `F()` + one `S(s)` per step, `s` following the real
  80-step production curve (`tanh(2t/80)/tanh(2)`), inter-step Aitken Δ²
  relaxation of the interface displacement (persisting `omega` across
  steps). This is a direct translation of a previously-validated
  algorithm into the F()/S(s) calling convention -- a fixed schedule, not
  an adaptive one.
- **Residual-adaptive pacing:** grow `s` faster when the previous step's
  residual (`mean_r` in the history) was small, slow down when it was
  large -- spend fewer `F()` calls when things are easy, more care when
  they're not, instead of a fixed curve.
- **Reusing a stale fluid state:** `F()` doesn't have to be called once
  per `S(s)` call -- if the wall hasn't moved much, an old fluid solve
  may still be "good enough" for the next solid step, saving an `F()`
  call (which is what efficiency actually charges for).
- **Backoff-and-retry:** on a solid-solve failure, instead of giving up
  immediately, retry with a smaller `s` (interpolating between the last
  successful value and the target) before advancing -- costs extra calls
  but may still reach `s_final == 1.0` (worth much more than the calls it
  costs, since it gates both stability and efficiency).

**What makes a good solution:**

- Reaches `s_final == 1.0` reliably (this gates both stability and
  efficiency credit) using as few `F()` calls as possible.
- Stays numerically close to the true converged displacement field --
  large deviations (even at `s_final == 1.0`) mean the schedule is
  cutting corners on accuracy to get there.

**Common pitfalls:**

- Reaching `s_final == 1.0` via a route that produces a badly diverged or
  unrealistic displacement field (e.g. huge jumps in `s`) will score well
  on stability/efficiency but badly on accuracy.
- Forgetting to write `weak_omega_history.json` after *every* successful
  `S(s)` call (not just the last) makes the run unscoreable if it's
  killed by the timeout partway through.
- `s` must be applied via `S(s)`'s own mechanism (a dynamically-extended
  load curve) -- an earlier version of this seed tried driving the target
  magnitude (`gr_insult.mag`) directly instead while forcing the
  material model's internal ramp to a constant. That desynced OTHER
  blended properties in the model from the intended schedule (they're
  also driven by that same internal ramp, not just the insult magnitude)
  and gave a different, wrong accuracy result (0.78 instead of the
  validated 0.62) despite reaching the same nominal `s_final`. `S(s)`
  already handles this correctly -- don't try to bypass it by touching
  `fsg.p["gr_insult"]` or the ramp mechanism yourself.

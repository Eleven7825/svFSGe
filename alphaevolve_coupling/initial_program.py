"""AlphaEvolve seed: fluid-structure-growth coupling built from two
black-box primitives, F() and S(s) -- call them in whatever pattern,
order, or frequency you want. What's being optimized is HOW to arrange
those calls (and what ramp fraction to drive S with at each call) so that
S reaches its final, fully-grown (s=1.0) solution as cheaply and
accurately as possible.

`run_coupling(fsg, t_start, i_start)` is monkey-patched onto the FSG class
as `_run_weak` by the evaluation harness (run_in_container.py) before
`fsg.run()` is called. `fsg` is a live FSG instance (see
~/dockers/svFSGe/fsg.py and svfsi.py) -- mesh generation, solver invocation,
and file I/O are all fixed infrastructure; F/S already wrap them for you.
Never call fsg.step(...) directly, never touch the C++ solver internals.

F() and S(s) -- defined below, outside the evolve-block (fixed, don't
reimplement):
  F()    One fluid(+mesh) solve on the current wall. Returns True on
         failure. Cheap to call repeatedly if you want to keep the fluid
         side fresh; also fine to skip and reuse a stale fluid state
         across multiple S() calls if you judge that safe.
  S(s)   One solid solve, applying ramp fraction `s` DIRECTLY: the
         material model's f_time (which governs how much of the target
         G&R state -- insult knockdown AND every other blended property
         that ramps alongside it -- is active this step) is driven by a
         load curve (fsg.p["gr_load"]) that gets extended with the
         (step, s) pair on every call, so f_time == s exactly at the step
         you just called. fsg.p["gr_insult"]["mag"] stays fixed at its
         configured target the whole run -- don't touch it; `s` is the
         only lever. `s` is unitless, meant to be compared to 1.0 (fully
         grown) for the stability score. Returns True on failure.

         (Why not just set gr_insult.mag = s*mag_target directly instead?
         Tried that first -- f_time also drives OTHER blended properties
         in the material model beyond insult magnitude, so forcing
         f_time to a constant while only scaling mag separately silently
         desyncs those other properties from your intended schedule.
         Confirmed empirically: that version reached the same final s=1.0
         but with meaningfully different accuracy (0.78 vs the validated
         0.62) than the real curve-driven mechanism. Driving f_time itself
         via the curve is what's actually validated to match.)

Both mutate fsg's internal state (fsg.curr/fsg.prev, solver call counters,
fsg.p["gr_load"]) and must be called in a sane order (a fresh run needs at
least one S(0.0) prestress call before any real growth is meaningful; F()
needs the mesh warped from a prior S() to have somewhere to solve on
beyond the very first call -- this is handled automatically, not something
to manage yourself).

Scoring reads two things back from disk (not from this function's return
value -- it returns nothing):
  1. weak_omega_history.json, written incrementally to fsg.p["f_out"]
     after every S() call that succeeds. Each entry MUST include "s": the
     value you called S with. Last entry's "s" is the stability score.
     Keep writing this file after every successful step, not only at the
     end: scoring is robust to this process being killed mid-run (e.g. by
     the 30-minute timeout) and only sees whatever was flushed before
     that.
  2. fsg.p["f_conv"] (== fsg.p["f_out"]/partitioned/converged), where
     fsg.save_tube(t, fsg.p["f_conv"]) writes one tube_<t>.vtu per
     successfully converged S() call -- the highest-numbered one is
     compared against the accuracy reference (a strong-coupling run at
     the same final target).

The score only cares about how far `s` got before something crashed, plus
the accuracy of the final state and how many F() calls it took -- not
which schedule or call pattern you used to get there. Explore.

Baseline behavior below: the same call pattern and Aitken relaxation
validated before this abstraction existed -- one F() + one S() per step,
`s` following the real 80-step production curve (tanh(2t/80)/tanh(2)) so
that s=1.0 at t=80 means true full load. This is deliberately just a
translation of the known-good baseline into the new F()/S() calling
convention, not a new algorithm; feel free to replace the schedule, the
call pattern, or the relaxation scheme entirely.
"""

import json
import math
import os

import numpy as np


def _make_primitives(fsg):
    """F() and S(s) -- see module docstring. Fixed infrastructure, not
    part of the evolve-block.

    t (the physics load-step index, distinct from i_f/i's file-naming
    counters) MUST come from S()'s own call count, not a counter shared
    with F(): the solid material model treats t==0 as the prestress phase
    specifically (gr_equilibrated.cpp checks 0 <= t <= pretime+epst). If
    F() bumped the same counter first, S()'s first (prestress) call would
    land on t=1+ instead of t=0 and the solid solve diverges (confirmed
    empirically: NaN residuals from iteration 1). F() itself doesn't need
    its own meaningful t (fmax=1.0 makes the only thing t affects on the
    fluid side, the pressure ramp p_vec[t], a constant regardless of t)
    -- it just reuses whatever t S() last used.
    """
    state = {"i_f": 0, "i_s": 0, "t": 0, "f_called": False}
    nloads = fsg.p["nloads"]
    curve = []   # accumulated (step, s) history fed to fsg.p["gr_load"]

    def F():
        state["i_f"] += 1
        i_f = state["i_f"]
        t = min(state["t"], nloads)
        times = {}
        if fsg.no is not None:
            fsg._neural_operator_step(times, i_f, t, 0)
        elif fsg.p["fsi"] and state["f_called"]:
            if fsg.step("mesh", i_f, t, 0, times):
                print("mesh simulation failed")
                return True
        state["f_called"] = True
        if fsg.no is None:
            if fsg.p["fsi"]:
                if fsg.step("fluid", i_f, t, 0, times):
                    print("fluid simulation failed")
                    return True
            else:
                fsg.poiseuille(t)
        return False

    def S(s):
        i = state["i_s"] + 1
        t = min(state["i_s"], nloads)   # 0 on S's first call (prestress)
        state["i_s"] = i
        state["t"] = t
        curve.append([t, s])
        fsg.p["gr_load"] = {"profile": "file", "curve": list(curve)}
        fsg.set_gr_load()
        times = {}
        if fsg.step("solid", i, t, 0, times):
            print("solid simulation failed")
            return True
        return False

    return F, S


# EVOLVE-BLOCK-START
def run_coupling(fsg, t_start, i_start):
    F, S = _make_primitives(fsg)

    ac        = fsg.p["coup"]
    omega0    = ac.get("omega0", 0.1)
    omega_min = ac.get("omega_min", 0.1)
    nloads    = fsg.p["nloads"]

    fsg.weak_history = []
    hist_path = os.path.join(fsg.p["f_out"], "weak_omega_history.json")

    def _save_history():
        with open(hist_path, "w") as f:
            json.dump(fsg.weak_history, f, indent=2)

    # ---- t == 0: prestress, no growth yet ----
    if F():
        fsg._save_failure_case(0, 0); return
    if S(0.0):
        fsg._save_failure_case(0, 0); return
    fsg.prev = fsg.curr.copy()
    print("  [seed] t0 prestress converged")
    fsg.err["disp"].append([1.0])  # placeholder: no prior state to compare
    fsg.save_tube(0, fsg.p["f_conv"])
    fsg.converged += [fsg.curr.copy()]
    fsg.weak_history.append({"t": 0, "mean_r": None, "omega": omega0, "s": 0.0})
    _save_history()

    omega    = omega0   # persists across steps (outer-loop history)
    res_prev = None

    for t in range(max(t_start, 1), nloads + 1):
        # real 80-step production load curve, s reaches 1.0 exactly at
        # t == nloads(=80): this is the ONLY place the schedule is
        # decided. Replace with anything -- residual-adaptive pacing,
        # backoff-and-retry, a totally different F/S call pattern.
        s_t = math.tanh(2 * t / 80) / math.tanh(2)

        print("=" * 30 + " t " + str(t) + " ==== s " + "{:.3f}".format(s_t) + " " + "=" * 30)

        wall     = fsg.curr.copy()                              # d_{t-1}
        wall_int = wall.get(("solid", "disp", "int")).flatten()

        if F():
            fsg._save_failure_case(t, t); return

        # single solid solve at s_t; n=0 -> always "beginning of new
        # load step" (restart from committed step-(t-1) state), since
        # there is no within-step sub-iteration to continue from.
        if S(s_t) or any(v is None for v in fsg.curr.sol.values()):
            print("  [seed] t%d: solid failed" % t)
            fsg._save_failure_case(t, t); return

        # step residual r_t = d_t* - d_{t-1}
        d_star = fsg.curr.get(("solid", "disp", "int")).flatten()
        r = d_star - wall_int

        # Aitken Delta^2 (Kuettler) update of omega from (r_{t-1}, r_t)
        if res_prev is not None:
            diff  = r - res_prev
            denom = float(np.dot(diff, diff))
            if denom > 0.0:
                omega = -omega * float(np.dot(res_prev, diff)) / denom
                omega = min(max(omega, omega_min), 1.0)
        res_prev = r

        # relaxed interface update d_t <- omega*d_t* + (1-omega)*d_{t-1}
        curr_v = fsg.curr.get(("solid", "disp", "vol"))    # d_t*
        prev_v = wall.get(("solid", "disp", "vol"))         # d_{t-1}
        fsg.curr.add(("solid", "disp", "vol"),
                      omega * curr_v + (1.0 - omega) * prev_v)

        err = float(np.mean(np.linalg.norm(r.reshape(-1, 3), axis=1)))
        print("  [seed] t%d: mean|r|=%.3e, omega=%.3f, s=%.3f" % (t, err, omega, s_t))

        fsg.weak_history.append({"t": t, "mean_r": err, "omega": omega, "s": s_t})
        _save_history()

        # also feed the standard self.err bookkeeping (one singleton
        # sub-iteration list per load step) so archive()/compare_results.py
        # work unmodified, same as the other coupling methods.
        fsg.err["disp"].append([err])

        fsg.save_tube(t, fsg.p["f_conv"])
        fsg.converged += [fsg.curr.copy()]
# EVOLVE-BLOCK-END

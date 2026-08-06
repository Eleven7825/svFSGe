"""AlphaEvolve seed: fluid-structure-growth weak coupling with an evolvable
insult-application schedule.

`run_weak_coupling(fsg, t_start, i_start)` is monkey-patched onto the FSG
class as `_run_weak` by the evaluation harness (run_in_container.py) before
`fsg.run()` is called, so it replaces today's `FSG._run_weak` for the
duration of one evaluation. `fsg` is a live FSG instance (see
~/dockers/svFSGe/fsg.py and svfsi.py) -- mesh generation, solver invocation
(`fsg.step(...)`), and file I/O are all fixed infrastructure and must not be
reimplemented here; only the orchestration logic below is yours to mutate.

Available on `fsg` (read, don't reinvent):
  - fsg.p               the run's config dict, notably:
      fsg.p["nloads"]        number of G&R load steps
      fsg.p["coup"]          coupling knobs (omega0, omega_min, ...)
      fsg.p["gr_load"]       temporal ramp: profile="file" with the real
                             80-step production load curve
                             (f(t) = tanh(2*t/80)/tanh(2), t=0..80; nloads
                             is 80 for this experiment, so t==nloads==80
                             reaches f(80)=1.0, true full load). The solid
                             solver applies this curve internally (see
                             gr_equilibrated.cpp's f_time) on top of
                             fsg.p["gr_insult"]["mag"].
      fsg.p["gr_insult"]     spatial insult profile (mag, z_loc, z_wid,
                             z_exp, asym, theta_wid, theta_exp) -- "mag" is
                             the peak magnitude that fsg.p["gr_load"]'s curve
                             (above) scales over pseudo-time; leave it
                             untouched unless you deliberately want to
                             override that built-in ramp (e.g. to retry a
                             failed step at a reduced scalar); this
                             function's job is to TRACK progress through
                             this run's step budget (as "s_t" below), not to
                             drive the insult level by default.
  - fsg.step(name, i, t, n, times)   name in {"fluid", "solid", "mesh"};
                             returns True on failure. The ONLY way to
                             advance the simulation -- never call the
                             solvers directly.
  - fsg.set_gr_insult()     re-patches the solid solver's input XML with
                             whatever is currently in fsg.p["gr_insult"].
                             Safe to call repeatedly with different "mag"
                             values between load steps -- but each call
                             STACKS with the solver's own internal ramp
                             (both independently go 0->1), so overriding
                             "mag" every step the way the naive baseline
                             once did double-ramps and produces a
                             different (and worse) loading history than
                             intended. Use deliberately, not routinely.
  - fsg.curr / fsg.prev     current / previous interface state accessors
                             (.get(("solid","disp","int")) etc.)
  - fsg._save_failure_case(t, i)   call this and return on any solver
                             failure (mesh/fluid/solid all failed already
                             logged their own reason; this just archives
                             the last-good state).

Scoring reads two things back from disk (not from this function's return
value -- it returns nothing):
  1. weak_omega_history.json, written incrementally to fsg.p["f_out"]
     after every load step this function successfully completes. Each
     entry MUST include "s": the fraction of THIS RUN'S step budget
     reached in [0, 1] (last entry's "s" == 1.0 means all nloads(=80)
     steps completed without failure -- true full load, since gr_load's
     curve runs the whole real 80-step trajectory -- the stability
     objective). Keep writing this file after every successful step, not
     only at the end: scoring is robust to this process being killed
     mid-run (e.g. by the 30-minute timeout) and only sees whatever was
     flushed before that.
  2. fsg.p["f_conv"] (== fsg.p["f_out"]/partitioned/converged), where
     fsg.save_tube(t, fsg.p["f_conv"]) writes one tube_<t>.vtu per
     successfully converged step -- the highest-numbered one is compared
     against the accuracy reference.

Baseline behavior below: track progress linearly with the load step,
s_t = t / nloads (reaching exactly 1.0 at t == nloads(=80), true full
load). This is deliberately the simplest possible tracking; feel free to
replace it with something that reacts to the residual, backs off and
retries at a smaller step on failure, or anything else that improves
stability/accuracy/efficiency.
"""

import json
import os

import numpy as np


# EVOLVE-BLOCK-START
def run_weak_coupling(fsg, t_start, i_start):
    ac        = fsg.p["coup"]
    omega0    = ac.get("omega0", 0.1)
    omega_min = ac.get("omega_min", 0.1)

    mag_target = fsg.p["gr_insult"]["mag"]
    nloads     = fsg.p["nloads"]

    i   = i_start   # solid file/log counter (monotonic)
    i_f = i_start   # mesh/fluid file counter

    def _fluid_solve(i_f, t):
        """One mesh+fluid (or NN) solve on the current wall. Returns True on
        failure."""
        times = {}
        if fsg.no is not None:
            fsg._neural_operator_step(times, i_f, t, 0)
        elif fsg.p["fsi"] and i_f > 1:
            if fsg.step("mesh", i_f, t, 0, times):
                print("mesh simulation failed"); return True, times
        if fsg.no is None:
            if fsg.p["fsi"]:
                if fsg.step("fluid", i_f, t, 0, times):
                    print("fluid simulation failed"); return True, times
            else:
                fsg.poiseuille(t)
        return False, times

    omega    = omega0   # persists across load steps (outer-loop history)
    res_prev = None

    # dedicated omega/residual/insult-scalar history (per load step t),
    # independent of the self.err/self.p["coup"]["omega"] bookkeeping used
    # by the other coupling methods (which this function never populates).
    # Written to f_out/weak_omega_history.json after every step so a
    # mid-run kill still leaves a complete record up to the last successful
    # step, including "s" (the insult scalar reached) for scoring.
    fsg.weak_history = []
    hist_path = os.path.join(fsg.p["f_out"], "weak_omega_history.json")

    def _save_weak_history():
        with open(hist_path, "w") as f:
            json.dump(fsg.weak_history, f, indent=2)

    for t in range(t_start, nloads + 1):
        # progress through THIS run's step budget: s_t == 1.0 exactly at
        # t == nloads(=80), meaning true full load reached (gr_load's
        # curve is the real 80-step production trajectory in full).
        # t == 0 naturally gives s_0 == 0.0 (prestress).
        s_t = t / nloads
        # NOTE: the solid solver already applies fsg.p["gr_load"]'s curve
        # internally over pseudo-time to scale fsg.p["gr_insult"]["mag"]
        # (see gr_equilibrated.cpp's f_time) -- that IS the real ramp.
        # Rewriting "mag" here on every step as well would double-ramp (an
        # additional independent 0->1 factor stacking on top of the
        # curve's own values, compounding into a different and wrong
        # loading history), so the baseline leaves "mag" untouched and
        # only *tracks* s_t for scoring. fsg.set_gr_insult() is still
        # available if you want the algorithm to actively drive the scale
        # itself (e.g. deliberately retrying a failed step at a lower
        # scalar before advancing) -- just be aware of what it's stacking
        # on top of if you do.

        print("=" * 30 + " t " + str(t) + " ==== fp "
              + "{:.2f}".format(fsg.p_vec[t]) + " ==== s "
              + "{:.3f}".format(s_t) + " " + "=" * 30)

        # ---- t == 0: prestress, one plain solid solve ----
        if t == 0:
            i_f += 1
            failed, times = _fluid_solve(i_f, t)
            if failed:
                fsg._save_failure_case(t, i); return
            fsg.prev = fsg.curr.copy()
            i += 1
            if fsg.step("solid", i, t, 0, times):
                print("solid simulation failed"); fsg._save_failure_case(t, i); return
            print("  [weak] t0 prestress converged")
            fsg.err["disp"].append([1.0])  # placeholder: no prior state to compare
            fsg.save_tube(t, fsg.p["f_conv"])
            fsg.converged += [fsg.curr.copy()]
            fsg.weak_history.append({"t": t, "mean_r": None, "omega": omega, "s": s_t})
            _save_weak_history()
            continue

        # ---- t > 0: ONE fluid + ONE solid solve, inter-step Aitken relaxation ----
        wall     = fsg.curr.copy()                              # d_{t-1}
        wall_int = wall.get(("solid", "disp", "int")).flatten()

        i_f += 1
        failed, times = _fluid_solve(i_f, t)
        if failed:
            fsg._save_failure_case(t, i); return

        # single solid solve; n=0 -> always "beginning of new load step"
        # (restart from committed step-(t-1) state), since there is no
        # within-step sub-iteration to continue from.
        i += 1
        times_s = {}
        failed = fsg.step("solid", i, t, 0, times_s)
        if failed or any(s is None for s in fsg.curr.sol.values()):
            print("  [weak] t%d: solid failed" % t)
            fsg._save_failure_case(t, i); return

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
        print("  [weak] t%d: mean|r|=%.3e, omega=%.3f, s=%.3f" % (t, err, omega, s_t))

        fsg.weak_history.append({"t": t, "mean_r": err, "omega": omega, "s": s_t})
        _save_weak_history()

        # also feed the standard self.err bookkeeping (one singleton
        # sub-iteration list per load step) so archive()/compare_results.py
        # work unmodified, same as the other coupling methods.
        fsg.err["disp"].append([err])

        fsg.save_tube(t, fsg.p["f_conv"])
        fsg.converged += [fsg.curr.copy()]
# EVOLVE-BLOCK-END

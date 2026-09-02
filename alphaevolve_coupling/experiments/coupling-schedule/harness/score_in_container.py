#!/usr/bin/env python3
"""Scores one AlphaEvolve weak-coupling evaluation from on-disk artifacts
left by run_in_container.py. Robust to that process having been killed by
a timeout mid-run: reads whatever partial state exists rather than
requiring a clean finish.

Runs inside the fsg-dev container (needs meshio/vtk to read the VTU
files, which are only guaranteed present there).

    python3 score_in_container.py --workdir <dir> --result <result.json path>

Score breakdown (weights fixed by the experiment design, not evolvable):
  stability  (0.4) = s_final in [0, 1], the last ramp fraction "s" the
                      candidate's own S(s) primitive successfully applied
                      before any failure (see weak_omega_history.json --
                      "s" is chosen freely by the evolved code, not tied
                      to a fixed schedule or step count). s_final == 1.0
                      means true full load (f_time=1.0) was reached.
  accuracy   (0.4) = Displacement-only comparison of the run's last
                      converged tube_*.vtu against a tight-tolerance
                      ground truth reaching the SAME final target (the
                      full static gr_insult.mag) via a different, faster
                      ramp pace (test_reference/vanilla_tolfinal_n10 --
                      see the REFERENCE_VTU note above for why comparing
                      across different paces to the same endpoint is
                      valid). Computed via exp(-k*err) with the CI's own
                      atol=rtol=1e-8 tolerance formula; k calibrated so a
                      uniform 5% relative error scores 0.65. Computed
                      unconditionally (no gating on s_final) -- a run that
                      didn't complete all nloads steps is compared against
                      the full-load reference as-is.
  efficiency (0.2) = 0 unless s_final == 1.0 exactly (no credit for a
                      partial run); otherwise a log-scaled function of
                      the number of fluid solves performed (cmin=3,
                      cmax=200).
"""
import argparse
import glob
import json
import math
import os

REFERENCE_VTU = "/svFSGe/test_reference/vanilla_tolfinal_n10/tube_010.vtu"
# Paced differently from this experiment's run (a 10-step self-scaled ramp
# vs. the real 80-step curve used here), but both reach the SAME final
# physical target: f_time=1.0, i.e. the full static gr_insult.mag applied.
# Comparing final states across different ramp paces is valid because
# what's being checked is the converged endpoint, not the trajectory to
# get there. Confirmed empirically: weak coupling's true full-load state
# (nloads=80) vs. this reference gives ~5% median relative displacement
# error -- night and day from comparing mismatched load levels (~65%
# error), which is what an earlier, buggy version of this scoring did.
ATOL = RTOL = 1.0e-8   # Displacement CI tolerance, scripts/compare_results.py
CMIN, CMAX = 3, 200     # efficiency cost-curve endpoints (fluid-solve count)


def find_f_out(workdir):
    """fsg.py writes into <workdir>/<config name>_<timestamp>/, created
    fresh by FSG.__init__. There is exactly one such subdirectory per
    evaluation (one config.json per run); if none exists, the run never
    got far enough to instantiate FSG at all."""
    candidates = [d for d in glob.glob(os.path.join(workdir, "*")) if os.path.isdir(d)]
    candidates = [d for d in candidates
                  if os.path.isdir(os.path.join(d, "partitioned"))
                  or os.path.isfile(os.path.join(d, "weak_omega_history.json"))]
    return sorted(candidates)[-1] if candidates else None


def load_s_final(f_out):
    if f_out is None:
        return 0.0
    hist_path = os.path.join(f_out, "weak_omega_history.json")
    if not os.path.isfile(hist_path):
        return 0.0
    with open(hist_path) as f:
        history = json.load(f)
    if not history:
        return 0.0
    return float(history[-1].get("s", 0.0))


def count_fluid_solves(f_out):
    if f_out is None:
        return 0
    return len(glob.glob(os.path.join(f_out, "partitioned", "fluid_*.log")))


def latest_converged_vtu(f_out):
    if f_out is None:
        return None
    matches = sorted(glob.glob(os.path.join(f_out, "partitioned", "converged", "tube_*.vtu")))
    return matches[-1] if matches else None


def displacement_accuracy(test_vtu_path):
    import meshio
    import numpy as np

    ref_disp = meshio.read(REFERENCE_VTU).point_data["Displacement"].astype(float).flatten()
    test_disp = meshio.read(test_vtu_path).point_data["Displacement"].astype(float).flatten()
    if test_disp.shape != ref_disp.shape:
        raise ValueError(f"shape mismatch: test={test_disp.shape} ref={ref_disp.shape}")

    ref_abs = np.abs(ref_disp)
    denom = ATOL + RTOL * ref_abs

    def mean_err(diff):
        return float(np.mean(np.maximum(0.0, (diff - ATOL - RTOL * ref_abs) / denom)))

    err = mean_err(np.abs(test_disp - ref_disp))

    # calibrate k from the same reference data: a uniform 5% relative
    # error (diff == 0.05*|ref|) should score exp(-k*err5) == 0.65
    err5 = mean_err(0.05 * ref_abs)
    k = -math.log(0.65) / err5 if err5 > 0 else 0.0

    return math.exp(-k * err)


def efficiency_score(cost):
    if cost <= 0:
        return 0.0
    raw = 1.0 - math.log(cost / CMIN) / math.log(CMAX / CMIN)
    return max(0.0, min(raw, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    f_out = find_f_out(args.workdir)

    s_final = load_s_final(f_out)
    stability = max(0.0, min(s_final, 1.0))

    fluid_count = count_fluid_solves(f_out)
    full_insult_reached = abs(s_final - 1.0) < 1e-9
    efficiency = efficiency_score(fluid_count) if full_insult_reached else 0.0

    insights = []
    vtu_path = latest_converged_vtu(f_out)
    if vtu_path is not None:
        try:
            accuracy = displacement_accuracy(vtu_path)
        except Exception as exc:
            accuracy = 0.0
            insights.append({"label": "accuracy_error", "text": str(exc)})
    else:
        accuracy = 0.0
        insights.append({"label": "no_state",
                          "text": "no converged tube_*.vtu found; run failed before "
                                   "completing any load step"})

    overall = 0.4 * stability + 0.4 * accuracy + 0.2 * efficiency

    insights.append({
        "label": "breakdown",
        "text": (f"s_final={s_final:.4f} stability={stability:.4f} "
                 f"accuracy={accuracy:.4f} fluid_solves={fluid_count} "
                 f"efficiency={efficiency:.4f} full_insult_reached={full_insult_reached}"),
    })

    with open(args.result, "w") as f:
        json.dump({"score": overall, "insights": insights}, f, indent=2)


if __name__ == "__main__":
    main()

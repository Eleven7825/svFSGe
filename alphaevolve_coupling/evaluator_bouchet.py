#!/usr/bin/env python3
"""AlphaEvolve evaluator for the svFSGe weak-coupling insult-scalar
experiment -- Bouchet (Yale YCRC HPC) variant.

Invoked by the ae CLI (running on Bouchet's login node) as:
    python evaluator_bouchet.py --output-file <path> --program-dir <path>
It execs initial_program.py found inside --program-dir.

Same contract and on-disk layout as evaluator.py (the local Docker
variant), but there is no persistent container here: each candidate's
simulation is dispatched to a compute node via a blocking `srun` +
`singularity exec` call instead of `docker exec`. This mirrors
docker_exec()'s two-step shape (run, then score) one-for-one -- see
evaluator.py's docstring for why the run/score split exists (the run step
can be killed by its own timeout without losing scoreable partial state).

Login-node etiquette: this process itself only polls the Cloud API and
blocks on srun calls -- it does no heavy compute of its own, so it's safe
to run for the AlphaEvolve experiment's full duration from the login node
(e.g. inside a `tmux` session). Every srun call requests its own
independent SLURM allocation; this must run from the login node (or any
shell with no existing job allocation of its own) -- if it ran inside a
job step already, `srun` would try to nest as a step within that job's
own allocation instead of getting independent resources.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

REPO_HOST = os.path.expanduser("~/svFSGe")
REPO_CONTAINER = "/svFSGe"
SVFSI_HOST = os.path.expanduser("~/svfsi")
IMAGE = os.path.join(REPO_HOST, "singularity_images", "simvascular-solver.sif")
EXPERIMENT_DIR = "alphaevolve_coupling"
HARNESS_DIR = f"{EXPERIMENT_DIR}/harness"

# SLURM resource request per evaluation -- mirrors the project's existing
# pulsatile.coarse.sbatch template (partition=day, 8 cores, 16G): fluid
# needs the most MPI ranks of the three solves (n_procs.fluid=3 in
# config_template.json), so 8 is already generous headroom, not a tight fit.
#
# --ntasks=1 --cpus-per-task=N, NOT --ntasks=N: srun runs the given command
# once per task, so --ntasks=N would launch N independent, colliding copies
# of the whole harness script instead of giving one process N cores to
# spawn its own internal mpiexec ranks on (confirmed empirically -- an
# earlier --ntasks=8 attempt ran 8 separate FSG() instances that stomped on
# each other's working directory). --ntasks=1 also keeps the allocation on
# a single node, which --ntasks=N alone does not guarantee.
SLURM_PARTITION = "day"
SLURM_TIME = "00:35:00"     # per-evaluation ceiling; comfortably above RUN_TIMEOUT_S
SLURM_CPUS = 8
SLURM_MEM = "16G"

RUN_TIMEOUT_S = 1750     # in-container watchdog; leaves margin under the 30 min cap
HOST_TIMEOUT_S = 1830    # includes SLURM queue wait, not just run time
SCORE_TIMEOUT_S = 300    # scoring is cheap; margin here is almost all queue wait


def srun_singularity(shell_command, timeout, cpus):
    singularity_cmd = (
        f"singularity exec --bind {SVFSI_HOST}:/svfsi --bind {REPO_HOST}:{REPO_CONTAINER} "
        f"{IMAGE} bash -lc {json.dumps(shell_command)}"
    )
    cmd = [
        "srun", f"--partition={SLURM_PARTITION}", f"--time={SLURM_TIME}",
        "--ntasks=1", f"--cpus-per-task={cpus}", f"--mem={SLURM_MEM}", "--job-name=ae-svfsge",
        "bash", "-lc", singularity_cmd,
    ]
    try:
        return subprocess.run(cmd, timeout=timeout, capture_output=True, text=True), None
    except subprocess.TimeoutExpired as exc:
        return None, exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--program-dir", required=True)
    args = parser.parse_args()

    program_path = os.path.join(args.program_dir, "initial_program.py")

    run_id = f"run_{int(time.time() * 1000)}_{os.getpid()}"
    host_workdir = os.path.join(REPO_HOST, EXPERIMENT_DIR, "runs", run_id)
    os.makedirs(host_workdir, exist_ok=True)
    container_workdir = f"{REPO_CONTAINER}/{EXPERIMENT_DIR}/runs/{run_id}"

    shutil.copy(program_path, os.path.join(host_workdir, "candidate_program.py"))
    shutil.copy(os.path.join(REPO_HOST, HARNESS_DIR, "config_template.json"),
                os.path.join(host_workdir, "config.json"))

    result_path_host = os.path.join(host_workdir, "result.json")
    result_path_container = f"{container_workdir}/result.json"

    insights = []

    run_cmd = (f"cd {container_workdir} && timeout {RUN_TIMEOUT_S} python3 "
               f"{REPO_CONTAINER}/{HARNESS_DIR}/run_in_container.py "
               f"--candidate {container_workdir}/candidate_program.py "
               f"--workdir {container_workdir}")
    run_result, run_timeout = srun_singularity(run_cmd, HOST_TIMEOUT_S, SLURM_CPUS)
    if run_timeout is not None:
        insights.append({"label": "timeout",
                          "text": "run exceeded the host-side subprocess timeout"})
    elif run_result.returncode != 0:
        tail = (run_result.stderr or run_result.stdout or "")[-2000:]
        insights.append({"label": "run_error",
                          "text": f"run_in_container.py exited {run_result.returncode}: {tail}"})

    score_cmd = (f"python3 {REPO_CONTAINER}/{HARNESS_DIR}/score_in_container.py "
                 f"--workdir {container_workdir} --result {result_path_container}")
    score_result, score_timeout = srun_singularity(score_cmd, SCORE_TIMEOUT_S, 1)
    if score_timeout is not None:
        insights.append({"label": "score_timeout", "text": "scoring step itself timed out"})
    elif score_result.returncode != 0:
        tail = (score_result.stderr or score_result.stdout or "")[-2000:]
        insights.append({"label": "score_error",
                          "text": f"score_in_container.py exited {score_result.returncode}: {tail}"})

    if os.path.isfile(result_path_host):
        with open(result_path_host) as f:
            result = json.load(f)
        result.setdefault("insights", []).extend(insights)
    else:
        result = {"score": 0.0,
                   "insights": insights + [{"label": "no_result",
                                             "text": "scoring step produced no result.json"}]}

    with open(args.output_file, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""AlphaEvolve evaluator for the svFSGe weak-coupling insult-scalar
experiment.

Invoked by the ae CLI as:
    python evaluator.py --output-file <path> --program-dir <path>
It execs initial_program.py found inside --program-dir.

The candidate can only run inside the fsg-dev Docker container -- the
actual svFSI solver binaries and MPI/PETSc environment live there, not on
whatever host/venv runs this evaluator. This process is a thin
orchestrator: copy the candidate + config template onto the shared
bind-mounted repo, `docker exec` the run (in-container `timeout` watchdog,
so the process tree is cleanly killed inside the container's own
namespace rather than orphaned), then `docker exec` the scoring step
(reads on-disk artifacts, robust to the run step having been killed).
"""
import argparse
import json
import os
import shutil
import subprocess
import time

REPO_HOST = "/home/shiyi/dockers/svFSGe"
REPO_CONTAINER = "/svFSGe"
EXPERIMENT_DIR = "alphaevolve_coupling"
# run_in_container.py/score_in_container.py/config_template.json live in a
# subdirectory, not alongside initial_program.py: the ae CLI bundles every
# top-level *.py file in --program-dir except evaluator.py/test_*/setup.py/
# conftest.py, so anything else there would get swept into the evolved
# program and mutated right along with it.
HARNESS_DIR = f"{EXPERIMENT_DIR}/harness"
CONTAINER = "fsg-dev"

RUN_TIMEOUT_S = 1750     # in-container watchdog; leaves margin under the 30 min cap
HOST_TIMEOUT_S = 1800    # hard host-side ceiling (matches the experiment's 30 min limit)
SCORE_TIMEOUT_S = 120    # scoring only reads files + one VTU comparison; should be fast


def docker_exec(container_workdir, shell_command, timeout):
    cmd = ["docker", "exec", "-w", container_workdir, CONTAINER, "bash", "-lc", shell_command]
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

    run_cmd = (f"timeout {RUN_TIMEOUT_S} python3 "
               f"{REPO_CONTAINER}/{HARNESS_DIR}/run_in_container.py "
               f"--candidate {container_workdir}/candidate_program.py "
               f"--workdir {container_workdir}")
    run_result, run_timeout = docker_exec(container_workdir, run_cmd, HOST_TIMEOUT_S)
    if run_timeout is not None:
        insights.append({"label": "timeout",
                          "text": "run exceeded the host-side subprocess timeout"})
    elif run_result.returncode != 0:
        tail = (run_result.stderr or run_result.stdout or "")[-2000:]
        insights.append({"label": "run_error",
                          "text": f"run_in_container.py exited {run_result.returncode}: {tail}"})

    score_cmd = (f"python3 {REPO_CONTAINER}/{HARNESS_DIR}/score_in_container.py "
                 f"--workdir {container_workdir} --result {result_path_container}")
    score_result, score_timeout = docker_exec(container_workdir, score_cmd, SCORE_TIMEOUT_S)
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

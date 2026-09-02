#!/usr/bin/env python3
"""Runs one AlphaEvolve candidate's weak-coupling algorithm.

Must run INSIDE the fsg-dev container: fsg.py's solver invocation uses
absolute in-container paths (/svfsi/svFSI-build/bin/svFSI) and the
MPI/PETSc environment that only exists there.

    python3 run_in_container.py --candidate <candidate_program.py> --workdir <dir>

--workdir must already contain config.json (written by the host-side
evaluator.py from config_template.json). All of fsg.py's normal run
artifacts (weak_omega_history.json, partitioned/converged/tube_*.vtu,
partitioned/fluid_*.log, ...) land under a fsg.py-generated subdirectory
of --workdir (named "<config name>_<timestamp>").

Scoring happens separately in score_in_container.py, reading those
artifacts back from disk -- kept as a separate process so a hard timeout
kill of THIS process still leaves scoreable partial output behind.
"""
import argparse
import importlib.util
import os
import sys
import traceback

REPO_ROOT = "/svFSGe"
sys.path.insert(0, REPO_ROOT)


def load_candidate(candidate_path):
    spec = importlib.util.spec_from_file_location("candidate_program", candidate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_coupling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    # On HPC deployments (e.g. Bouchet), this process runs inside a SLURM
    # job step; svFSI's mpiexec/prterun launcher auto-detects SLURM_* env
    # vars and sizes its process "slots" from SLURM_NTASKS rather than the
    # job's actual CPU allocation, refusing to start multi-rank solves
    # ("not enough slots") -- confirmed empirically. Stripping them here,
    # in Python, before any subprocess is spawned, is a no-op locally
    # (Docker has no SLURM env at all) and avoids the fragility of trying
    # to do this via a shell "unset" wrapper threaded through several
    # nested layers of shell quoting (srun -> singularity exec -> bash).
    for k in [k for k in os.environ if k.startswith("SLURM_")]:
        del os.environ[k]

    os.chdir(args.workdir)

    try:
        run_coupling = load_candidate(args.candidate)

        from fsg import FSG
        FSG._run_weak = run_coupling

        fsg = FSG(os.path.join(args.workdir, "config.json"))
        fsg.run()
    except Exception:
        # Left for the run's own stdout/stderr capture (the host evaluator
        # attaches it as a diagnostic insight); scoring reads whatever
        # on-disk artifacts exist regardless of how far this got.
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

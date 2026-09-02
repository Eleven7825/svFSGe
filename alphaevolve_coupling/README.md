# AlphaEvolve experiments for svFSGe

Evolves `run_coupling` (and, potentially, future FSGe problems) using Google
Cloud's AlphaEvolve.

## Layout

```
alphaevolve_coupling/
├── experiments/              one self-contained bundle per experiment DEFINITION
│   └── coupling-schedule/        evolves run_coupling's fluid-solid schedule
│       ├── problem_description.md   the problem statement given to the LLM
│       ├── initial_program.py       the seed program (bundled as the initial candidate)
│       ├── evaluator.py             local-Docker evaluator (fsg-dev container)
│       ├── harness/                 fixed infrastructure, never mutated by evolution
│       │   ├── config_template.json     simulation config for every candidate's run
│       │   ├── run_in_container.py      execs a candidate's run_coupling
│       │   ├── score_in_container.py    reads back the run's artifacts, computes score
│       │   └── evaluator_bouchet.py     Bouchet (Yale HPC) evaluator: dispatches via
│       │                                 srun+singularity instead of docker exec
│       └── runs/                    gitignored scratch — one dir per in-progress
│                                     evaluation's working files, safe to delete anytime
├── results/                  permanent archive, one dir per COMPLETED experiment RUN
│                              (see results/README.md for the convention, and for
│                              the difference between "experiment" and "result")
└── docs/                     harness design/validation notes, not experiment results
```

Each `experiments/<name>/` is a fully self-contained, independently
relocatable bundle — `evaluator.py`/`evaluator_bouchet.py` resolve their
harness siblings and scratch `runs/` relative to their own file location
(not a hardcoded shared path), so a new experiment idea is just a new sibling
directory under `experiments/`, with its own problem description, seed, and
evaluator/scoring logic, never touching an existing one.

`evaluator.py` and `evaluator_bouchet.py` are two different *dispatch*
backends for the exact same `run_in_container.py`/`score_in_container.py`
contract — local Docker vs. Bouchet's `srun`+`singularity`. Both bundle only
`initial_program.py` as the candidate program (the `ae` CLI excludes
`evaluator.py` from `--program-dir` bundling by name, and `harness/` is a
subdirectory so its non-recursive glob never picks up anything inside it).

## Running an experiment

All commands below run from inside the specific experiment's own directory
(e.g. `experiments/coupling-schedule/`), so `--program-dir .` and the
relative `--evaluator`/`--problem-file` paths resolve correctly.

Local (Docker, `fsg-dev` container):
```bash
cd experiments/coupling-schedule
ae --json program evaluate --evaluator evaluator.py --program-dir . --backend local
```

Bouchet (from the login node, inside a `tmux` session so it survives
disconnects):
```bash
cd experiments/coupling-schedule   # on Bouchet's own checkout
ae experiment create --max-programs <N> --concurrency <N> \
  --problem-file problem_description.md --title "<title>"
ae experiment start <experiment> --program-dir . --score <baseline_score>
ae experiment run <experiment> --evaluator harness/evaluator_bouchet.py \
  --backend local --dashboard runs/<experiment>-dashboard.md
```
Bouchet's `interactive` QOS caps concurrent `srun` jobs at 4 regardless of
the experiment's configured concurrency — see `evaluator_bouchet.py`'s
docstring. To actually reach higher parallelism, launch multiple
`ae experiment run` processes against the same experiment (the backend's
lock-token acquire mechanism is designed for exactly this).

When an experiment run finishes, archive it under `results/<nickname>/` —
see `results/README.md`.

## Starting a new experiment

Create a new sibling under `experiments/`, e.g. `experiments/<new-idea>/`,
with its own `problem_description.md` + `initial_program.py`. Reuse
`coupling-schedule`'s `harness/`+`evaluator.py`/`evaluator_bouchet.py` as a
starting template if the new experiment is still evaluating `run_coupling`
via the same `fsg.py`/`svfsi.py` machinery — copy the whole directory and
edit `run_in_container.py`/`score_in_container.py`/`config_template.json`
as needed for what's actually different (a different scoring metric, a
different monkey-patched entry point, etc.).

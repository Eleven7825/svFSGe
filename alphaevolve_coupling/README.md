# AlphaEvolve: evolving the FSGe coupling schedule

Evolves `run_coupling` — the fluid-solid coupling schedule built from the
`F()`/`S(s)` black-box primitives (see `initial_program.py`'s docstring for
the full problem statement) — using Google Cloud's AlphaEvolve.

## Layout

```
alphaevolve_coupling/
├── problem_description.md   the problem statement given to the LLM
├── initial_program.py       the seed program (bundled as the initial candidate)
├── evaluator.py             local-Docker evaluator (fsg-dev container)
├── harness/                 fixed infrastructure, never mutated by evolution
│   ├── config_template.json     simulation config for every candidate's run
│   ├── run_in_container.py      execs a candidate's run_coupling
│   ├── score_in_container.py    reads back the run's artifacts, computes score
│   └── evaluator_bouchet.py     Bouchet (Yale HPC) evaluator: dispatches via
│                                 srun+singularity instead of docker exec
├── runs/                    gitignored scratch — one dir per in-progress
│                             evaluation's working files, safe to delete anytime
├── results/                 permanent archive, one dir per COMPLETED experiment
│                             (see results/README.md for the convention)
└── docs/                    harness design/validation notes, not experiment results
```

`evaluator.py` and `evaluator_bouchet.py` are two different *dispatch*
backends for the exact same `run_in_container.py`/`score_in_container.py`
contract — local Docker vs. Bouchet's `srun`+`singularity`. Both bundle only
`initial_program.py` as the candidate program (the `ae` CLI excludes
`evaluator.py` from `--program-dir` bundling by name, and `harness/` is a
subdirectory so its non-recursive glob never picks up anything inside it).

## Running an experiment

Local (Docker, `fsg-dev` container):
```bash
ae --json program evaluate --evaluator evaluator.py --program-dir . --backend local
```

Bouchet (from the login node, inside a `tmux` session so it survives
disconnects):
```bash
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

When an experiment finishes, archive it under `results/<nickname>/` — see
`results/README.md`.

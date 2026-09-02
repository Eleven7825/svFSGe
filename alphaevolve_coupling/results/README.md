# Experiment results archive

One subdirectory per **completed** AlphaEvolve experiment, named by its `ae`
nickname (e.g. `exp-steadfast-bullfinch`). Archived here so a run's full
history survives independently of the Bouchet backend (which can be deleted
via `ae experiment delete` with no undo) and doesn't require SSHing into
Bouchet to look up later.

## Convention for archiving a new experiment

Once an experiment reaches a terminal state (`COMPLETED`/`FAILED`), create
`results/<experiment-nickname>/` containing:

- `programs.json` — raw dump of every program in the experiment:
  `ae --json program list <experiment> > programs.json`. This is the
  canonical record; everything else here can be regenerated from it.
- `README.md` — a short summary: run parameters (concurrency, max-programs,
  backend), final stats (total/completed/crashed, best score vs. baseline),
  the winning lineage with what each step changed, and any caveats that
  affect how to interpret the results (e.g. a leaked hint in the problem
  description, a scoring bug fixed partway through).
- Any visualizations built from `programs.json` (e.g. a lineage-tree HTML
  like `exp-steadfast-bullfinch/coupling_genealogy.html`) — self-contained,
  openable directly in a browser, so they don't depend on a Claude Artifact
  or any other external host staying up.

`runs/` (sibling directory, gitignored) is the opposite of this: a
*transient* scratch area for in-progress evaluation working directories
during an active experiment. Nothing under `runs/` is meant to persist —
once an experiment is done, the parts worth keeping get archived here
instead.

## Experiments archived so far

- [`exp-steadfast-bullfinch/`](exp-steadfast-bullfinch/README.md) — the
  production run evolving `run_coupling`'s fluid-solid coupling schedule.
  107 programs, best score 0.8314 (seed baseline 0.6924, +20.1%). The
  winning lineage (10-step compressed schedule with real sub-iteration) was
  ported back to `svFSGe`'s `master` branch as `fsg.py`'s `_run_uber_robin`.

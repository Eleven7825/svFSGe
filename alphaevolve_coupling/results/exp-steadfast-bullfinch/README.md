# exp-steadfast-bullfinch — AlphaEvolve production run

Full archive of the production AlphaEvolve experiment evolving `run_coupling`
(the fluid-solid coupling schedule for `svFSGe`'s F()/S(s) primitives). Saved
here so the run's history is available without SSHing into Bouchet or
depending on the backend experiment still existing.

This run was defined by
[`experiments/coupling-schedule/`](../../experiments/coupling-schedule/) —
its `problem_description.md`/`initial_program.py`/`evaluator.py` are the
source of truth for what was actually evolved; this directory is only the
archived *output* of running that definition once.

## Files

- `programs.json` — raw dump of every program in the experiment
  (`ae --json program list exp-steadfast-bullfinch`). One entry per program:
  full source code, score + breakdown insight, `parentPrograms` (lineage),
  state, and (for crashed programs) the run/score error traceback. This is
  the canonical record everything else can be regenerated from.
- `coupling_genealogy.html` — the interactive lineage-tree visualization
  built from `programs.json` (self-contained, open directly in a browser).
  Generation columns from the seed outward, winning lineage highlighted,
  hover for the model's own code comments, click a node for its full
  source. Also published as a Claude Artifact:
  https://claude.ai/code/artifact/cf775c0a-c792-472c-93b0-a0f449fd73d2

## Run parameters

- Concurrency: 10 (throttled to 4 concurrent SLURM jobs by Bouchet's
  `interactive` QOS job-count limit — see git history / conversation notes
  on `evaluator_bouchet.py` for the `srun`-vs-`sbatch` tradeoff)
- Max programs: 100 (actual: 107 — the parallel clients each grabbed one
  more before the terminal state was enforced)
- Backend: `srun` + `singularity`, dispatched via `evaluator_bouchet.py`

## Results

| | |
|---|---|
| Total programs | 107 |
| Completed | 105 (2 left orphaned `EVALUATING`, harmless) |
| Scored (non-crash) | 86 |
| Crashed | 19 (~18%, mostly a recurring `UnboundLocalError` from a redundant `import os` inside `run_coupling`, plus some `score_timeout` stragglers from before the scoring-dispatch fix) |
| Seed baseline | `prog-inventive-chachalaca`, score **0.6924** |
| Best found | `prog-denim-guan`, score **0.8314** (+20.1%) |

## The winning lineage

`prog-inventive-chachalaca` (seed, 0.6924)
→ `prog-uber-robin` (0.8222) — restructured from the seed's fixed 80-step
  schedule to 10 compressed steps, each genuinely sub-iterated to a fixed
  point (2 passes for intermediate steps, 6 for the final step) instead of
  advancing the load on every solve with no sub-iteration at all.
→ `prog-pumpkin-jackal` (0.8229) — added linear extrapolation warm-start
  between steps; fixed `omega` to 1.0 on genuine convergence instead of
  diluting an already-converged solution.
→ `prog-wine-emu` (0.8280) — graded the per-step sub-iteration tolerance
  (loose early, very tight at the final step) instead of one flat value.
→ `prog-denim-guan` (0.8314) — upgraded to 3-point Lagrange (quadratic)
  extrapolation; persists `omega` across outer steps instead of resetting
  it every step.

This lineage's core idea (`prog-uber-robin`'s restructuring) was ported
back into `svFSGe`'s `main` branch as `fsg.py`'s `_run_uber_robin` /
`coup.method="uber_robin"` — see `docs/coupling_algorithms.tex` Section 4
for the algorithm derivation and the `master` branch commit for the
implementation, validated to reproduce the exact same fluid-solve count
(25) as the original evolved candidate.

## Known caveat: the reference step count leaked into the search

`problem_description.md` explicitly named the accuracy reference's step
count (`test_reference/vanilla_tolfinal_n10`, "a faster 10-step ramp") —
this is visible to the LLM on every generation call. Roughly 44% of the
107 generated programs (47/107) reference "n10"/"10-step"/"reference" in
their own code comments, and the entire winning lineage above uses
`N=10`. Two independent branches did try `N=8` instead
(`prog-influential-hedgehog`, `prog-inescapable-lynx`) — one scored 0.508
(a real, fairly evaluated result, just lower), the other crashed before
producing a score. One descendant (`prog-notorious-lynx`) explicitly
reverted its own `N=8` back to `N=10`, citing the reference match by name
in its own comment. So the *specific* choice of N=10 in this run's
winning lineage is confounded by that leak, even though the sub-iteration
scheme, extrapolation, and relaxation-persistence refinements built on
top of it are genuine, independently-discovered improvements.

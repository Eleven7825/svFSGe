# Harness design notes

Supporting artifacts from designing/validating the evaluation harness itself
— not tied to any one experiment's results (those live under `../results/`).

- `displacement_comparison_fullload.png` — the sanity check behind the
  accuracy metric in `score_in_container.py`: strong coupling (a fast
  10-step ramp to full load) vs. weak coupling (the real 80-step ramp),
  both at their true full-load endpoint. Confirms that comparing final
  states across different ramp *paces* to the same physical target is
  valid — the two populations of points overlap almost exactly.

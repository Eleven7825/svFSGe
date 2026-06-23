# Configuration files for FSGe
Contains an example for an FSGe configuration file
### Parameters of configuration file
- `fsi` true (run fluid simulation), false (approximate Poiseuille solution)
- `debug` if true, prints `svFSIplus` simulation output to screen
- `mesh` geometry input file from `in_geo`
- `n_procs` number of processors to use for `fluid`, `mesh`, and `solid` simulations
- `fluid` parameters for Poiseuille flow solution
- `n_max` number of time steps in `fluid`, `mesh`, and `solid` simulations
- `coup` coupling parameters
  - `nmax` maximum number of coupling iterations
  - `nmin` minimum number of coupling iterations
  - `tol` coupling tolerance for convergence
  - `method` coupling method: `iqn_ils` (recommended), `static`, `aitken` (slow, likely unstable)
  - `omega0` damping parameter for static relaxation
  - `iqn_ils_q` IQN-ILS number of old iterations to use
  - `iqn_ils_eps` IQN-ILS filtering tolerance
- `nloads` number of G&R load steps (plus one pre-loading step). *(Renamed from
  `nmax`, which clashed with `coup.nmax`; the old name is still accepted.)*
- `gr_load` *(optional)* shape of how the G&R insult is ramped over the load steps.
  Previously this profile was hard-coded inside `gr_equilibrated.cpp`; it is now
  injected into the solid input file so it can be changed without rebuilding `svFSIplus`.
  Omit this section to keep the historical default (`tanh`, `steep` = 2.0).
  - `profile` ramp shape: `linear`, `tanh` (front-loaded, default), `power`, or `file`
  - `steep` shape parameter — `tanh` steepness or `power` exponent (default 2.0)
  - `curve` only for `profile: "file"` — the load factor **per step**, as a list of
    `[step, factor]` pairs (or a plain list of factors). `step` is the integer load-step
    number: `0` = pre-stress, `1..nloads` = G&R loads, so the curve has `nloads + 1`
    entries — one per step, no interpolation between steps. Example for `nloads = 3`:
    `[[0, 0.0], [1, 0.33], [2, 0.67], [3, 1.0]]`
- `exe` executable file paths for `svFSIplus` for `fluid`, `mesh`, and `solid` simulations
- `inp` input files for `svFSIplus` for `fluid`, `mesh`, and `solid` simulations
- `interfaces` names for various surfaces and input files for `fluid`, `mesh`, and `solid` simulations (should not be changed)
- `out` name of output folders for `fluid`, `mesh`, and `solid` simulations (must match with `svFSIplus` input files)
- `name` name of the FSGe simulation
- `paths_*` folder names for various operating systems
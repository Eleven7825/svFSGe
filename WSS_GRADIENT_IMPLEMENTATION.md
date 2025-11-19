# WSS Gradient Implementation for FSGe Growth Stimulus

## Problem Context

The FSGe (Fluid-Solid-Growth equilibrated) code models blood vessel growth and remodeling in response to mechanical stimuli. Originally, the growth stimulus was the **wall shear stress (WSS) magnitude** (||τ_w||). The goal is to change this to the **WSS gradient along the vessel** (∂WSS/∂z).

### Scientific Background

From the paper (Pfaller et al., 2024):
- The model uses mechanobiological equilibrium where tissue adapts to maintain homeostatic stress
- Two key stimuli drive growth:
  - **Intramural stress (IMS)**: σ_I (induced by pressure and axial force)
  - **Wall shear stress (WSS)**: τ_w (induced by blood flow)
- The equilibrium condition (Equation 12): Δσ_Ih = K_τσ · Δτ_wh

Where:
- Δσ_Ih = (σ_Ih / σ_Io) - 1 (intramural stress deviation)
- Δτ_wh = (τ_wh / τ_wo) - 1 (WSS deviation)
- K_τσ = shear-to-intramural gain ratio

## Goal

**Change the growth stimulus from WSS magnitude to WSS spatial gradient:**
- **Old**: τ_w = ||WSS vector||
- **New**: τ_w = ∂||WSS||/∂z (gradient along vessel axis)

## Implementation Overview

### Key Insight

The C++ growth formulation code (`gr_equilibrated.cpp`) is **generic** with respect to the stimulus. It uses:
```cpp
tau_ratio = tau / tauo;  // Line 620
```

Where:
- `tau` = current stimulus value (retrieved from `eVWP(6)`)
- `tauo` = homeostatic stimulus value (stored in `grInt(3)`)

By changing what we pass in `props[:, 6]` from WSS magnitude to WSS gradient, the entire formulation automatically uses gradient without C++ modifications!

### Implementation Details

#### 1. Mesh Structure (Cylinder Generation)

The solid mesh is **structured** with:
- **Configuration file**: `in_geo/fsg_full_coarse.json`
- Parameters:
  - `n_cir = 32`: circumferential points
  - `n_rad_gr = 1`: radial growth layers
  - `n_axi = 20`: axial slices
- **Points per z-slice**: `n_points_per_slice = n_cir × (n_rad_gr + 1) = 64`
- **Total points**: 1344 (for 21 z-slices including boundaries)

**Point indexing**:
- Points 0-63: z-slice 0 (inlet)
- Points 64-127: z-slice 1
- Points i and i+64: axial neighbors at same (r, θ) position

#### 2. Python Implementation (`svfsi.py`)

**Location**: Lines 309-356, 358-383

**New Function**: `compute_axial_wss_gradient(wss_magnitude)`

```python
def compute_axial_wss_gradient(self, wss_magnitude):
    """
    Compute axial (z-direction) gradient of WSS magnitude using finite differences.

    Uses structured mesh where points are organized as slices perpendicular to z-axis.
    Each slice has n_points_per_slice = n_cir * (n_rad_gr + 1) points.
    """
    # Get mesh parameters directly from mesh config
    n_cir = self.mesh_p["n_cir"]
    n_rad_gr = self.mesh_p["n_rad_gr"]
    n_points_per_slice = n_cir * (n_rad_gr + 1)

    points = self.points[("vol", "solid")]
    n_points = len(points)
    wss_gradient = np.zeros(n_points)

    for i in range(n_points):
        idx_prev = i - n_points_per_slice  # previous z-slice
        idx_next = i + n_points_per_slice  # next z-slice

        if idx_prev >= 0 and idx_next < n_points:
            # Interior: central difference
            dz = points[idx_next, 2] - points[idx_prev, 2]
            dwss = wss_magnitude[idx_next] - wss_magnitude[idx_prev]
            wss_gradient[i] = dwss / dz

        elif idx_prev < 0:
            # Boundary at z=0: forward difference
            dz = points[idx_next, 2] - points[i, 2]
            dwss = wss_magnitude[idx_next] - wss_magnitude[i]
            wss_gradient[i] = dwss / dz

        else:  # idx_next >= n_points
            # Boundary at z=end: backward difference
            dz = points[i, 2] - points[idx_prev, 2]
            dwss = wss_magnitude[i] - wss_magnitude[idx_prev]
            wss_gradient[i] = dwss / dz

    return wss_gradient
```

**Modified Function**: `set_solid(n, t)` (Lines 358-387)

Changes:
```python
# OLD (Line 318):
props[:, 6] = self.curr.get(("solid", "wss", "vol"))  # WSS magnitude

# NEW (Lines 369-375):
wss_mag = self.curr.get(("solid", "wss", "vol"))
wss_grad = self.compute_axial_wss_gradient(wss_mag)
props[:, 6] = wss_grad  # WSS gradient
```

#### 3. C++ Code (No Changes Required!)

**File**: `gr_equilibrated.cpp` (in archived simulation results)

**Key lines**:
- Line 128: `const double tau = eVWP(6);` - Retrieves stimulus
- Line 620: `tau_ratio = tau / tauo;` - Computes ratio
- Line 926: `grInt(3) = tau;` - Stores homeostatic value (during prestressing)

The code automatically works with gradient because:
1. During **prestressing** (t=0): stores current WSS gradient as homeostatic reference
2. During **growth** (t>0): computes ratio of current/homeostatic WSS gradient

## Data Flow

### 1. Fluid Simulation → WSS Extraction

**File**: `svfsi.py`, lines 436-461

```python
def post(self, domain, i):
    # ... (solve Navier-Stokes)

    if f == "wss":
        # Extract WSS vector from fluid solution
        sol = v2n(c2p.GetPointData().GetArray("WSS"))  # [N, 3] vector

        # Compute magnitude at interface
        self.curr.add((phys, f, "int"), sol[map_int])
```

**File**: `svfsi.py`, lines 676-689

```python
def add(self, kind, sol):
    # ...
    elif "wss" in f:
        # Store WSS magnitude in solution vector
        self.sol[f][map_v] = deepcopy(np.linalg.norm(sol, axis=1))

        # Propagate from interface to volume
        # (assumes radially uniform WSS)
        map_src = self.sim.map((("vol", "solid"), ("int", "fluid")))
        map_trg = self.sim.map((("vol", "solid"), ("vol", "tube")))
        self.sol[f][map_trg] = deepcopy(sol_int[map_src])
```

### 2. WSS Magnitude → WSS Gradient

**File**: `svfsi.py`, lines 358-383

Called every coupling iteration before solid solve:
```python
def set_solid(self, n, t):
    # Get WSS magnitude from current solution
    wss_mag = self.curr.get(("solid", "wss", "vol"))

    # Compute gradient
    wss_grad = self.compute_axial_wss_gradient(wss_mag)

    # Pass to solid solver via gr_properties array
    props = v2n(solid.GetPointData().GetArray("gr_properties"))
    props[:, 6] = wss_grad
```

### 3. Solid Solver → Growth Response

**File**: `gr_equilibrated.cpp`

```cpp
// Retrieve from input (line 128-131)
const double tau = eVWP(6);  // Now contains WSS gradient!
const vec3d dtau(eVWP(9), eVWP(10), eVWP(11));  // Gradient vector (unused)

// Compute ratio (line 619-622)
if (grM.coup_wss)
    tau_ratio = tau / tauo;  // gradient_current / gradient_homeostatic

// Use in growth equations (line 629, 646)
const double Cratio = CB - CS * (EPS * tau_ratio - 1.0);  // Active stress
p_gp = svh - svo/(1.0-delta) * (1.0 + KsKi * (EPS * tau_ratio - 1.0) - KfKi * inflam);
```

## Program Structure

```
svFSGe/
├── fsg.py                     # Main coupling driver (FSG class)
│   └── main()                 # Partitioned FSGe coupling loop
│       ├── coup_step_iqn_ils()  # Strong coupling with IQN-ILS
│       └── step()             # Individual domain solves
│
├── svfsi.py                   # Base simulation class (svFSI class)
│   ├── __init__()            # Initialize meshes and parameters
│   ├── set_fluid()           # Prepare fluid boundary conditions
│   ├── set_mesh()            # Prepare ALE mesh motion
│   ├── set_solid()           # Prepare solid with WSS gradient ← MODIFIED
│   ├── compute_axial_wss_gradient()  # Compute ∂WSS/∂z ← NEW
│   ├── step()                # Execute svFSIplus solver
│   └── post()                # Extract and process results
│
├── in_sim/
│   └── partitioned_full.json  # Simulation configuration
│       └── "mesh": "fsg_full_coarse.json"
│
├── in_geo/
│   └── fsg_full_coarse.json   # Mesh generation parameters
│       ├── n_cir: 32          # Circumferential points
│       ├── n_rad_gr: 1        # Radial growth layers
│       └── n_axi: 20          # Axial divisions
│
├── in_svfsi_plus/             # svFSIplus XML input files
│   ├── steady_full.xml        # Fluid solver config
│   ├── gr_full_restart.xml    # Solid/growth solver config
│   └── mesh_full.xml          # Mesh motion config
│
└── in_petsc/                  # PETSc linear solver settings
    └── direct.inp             # MUMPS direct solver
```

### gr_properties Array Structure

**Array shape**: `(n_points, 50)`

**Relevant indices**:
- `[0]`: Jo (original Jacobian)
- `[1]`: svo (original intramural stress)
- `[3]`: tauo (original WSS - **now gradient!**)
- `[6]`: tau (current WSS - **now gradient!**) ← Modified
- `[7]`: time step
- `[12]`: beginning of load step flag
- `[9-11]`: WSS gradient components (available but unused)

## Entry Points and Usage

### Running a Simulation

```bash
cd /svfsi/svFSGe

# Basic run
./fsg.py in_sim/partitioned_full.json

# Post-processing only
./fsg.py in_sim/partitioned_full.json -post
```

### Configuration Hierarchy

1. **Simulation config** (`in_sim/partitioned_full.json`):
   - References mesh config file
   - Sets coupling parameters (IQN-ILS, tolerances)
   - Sets fluid properties (viscosity, density)
   - Defines number of load steps

2. **Mesh config** (`in_geo/fsg_full_coarse.json`):
   - Defines geometry (radius, height)
   - Sets discretization (n_cir, n_rad_gr, n_axi)
   - These parameters are used by `compute_axial_wss_gradient()`

3. **Solver configs** (`in_svfsi_plus/*.xml`):
   - Finite element settings
   - Material model parameters (K_τσ gain ratio)
   - Boundary conditions

## Validation Plan

### 1. Check Gradient Calculation

```python
# In post-processing, compare:
# - WSS magnitude field
# - Computed WSS gradient field
# - Manually computed gradient (using VTK gradient filter)

# Verify:
# - Gradients are reasonable (not infinite/NaN)
# - Boundary conditions (forward/backward differences work)
# - Interior points (central difference more accurate)
```

### 2. Compare Growth Behavior

Run simulations with:
- **Old**: WSS magnitude stimulus
- **New**: WSS gradient stimulus

Compare:
- Aneurysm growth patterns
- Collagen mass deposition
- Radial displacements
- Convergence behavior

### 3. Check Homeostatic State

For uniform flow (no aneurysm):
- WSS gradient should be ≈ 0
- Vessel should remain stable (no growth)
- Verify prestressing still works correctly

## Known Limitations and Assumptions

1. **Structured mesh requirement**: The gradient calculation assumes a structured cylindrical mesh with constant `n_points_per_slice`

2. **Axial gradient only**: Computes ∂WSS/∂z, ignores circumferential and radial gradients

3. **Finite difference accuracy**:
   - Interior points: 2nd order accurate (central difference)
   - Boundary points: 1st order accurate (forward/backward difference)
   - Variable spacing may affect accuracy

4. **Radial uniformity**: Assumes WSS stimulus affects all radial layers equally (current formulation)

## Future Enhancements

1. **3D gradient**: Compute full gradient vector (∂WSS/∂x, ∂WSS/∂y, ∂WSS/∂z)
2. **Higher-order schemes**: Use more neighbor points for better accuracy
3. **Unstructured meshes**: Generalize using VTK gradient filters
4. **Directional stimulus**: Use gradient magnitude and direction separately

## References

- Paper: Pfaller et al. (2024) "FSGe: A fast and strongly-coupled 3D fluid–solid-growth interaction method"
- Location: `/svfsi/svFSGe/Pfaller et al. - 2024 - FSGe A fast and strongly-coupled 3D fluid–solid-growth interaction method.pdf`
- Key equations: Equations 11, 12, 15 (pages 4-5)

## Implementation Date

November 3, 2025

## Modified Files

1. **svfsi.py** (lines 309-383):
   - Added: `compute_axial_wss_gradient()` method
   - Modified: `set_solid()` to compute and pass WSS gradient

2. **This documentation**: `WSS_GRADIENT_IMPLEMENTATION.md`

## No Changes Required

- **gr_equilibrated.cpp**: Works generically with any stimulus
- **fsg.py**: Coupling logic unchanged
- **Configuration files**: No changes needed
- **Mesh generation**: No changes needed

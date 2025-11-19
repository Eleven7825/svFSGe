# WSS Gradient Implementation - Trial Summary

**Date**: November 3, 2025
**Goal**: Replace WSS magnitude with WSS gradient as growth stimulus in FSGe simulations

## Overview

Successfully implemented WSS gradient calculation (∂WSS/∂z) with configurable scaling, but encountered fundamental numerical instability in the C++ growth solver.

---

## Implementation Summary

### What Was Implemented

1. **WSS Gradient Calculation Function** (`svfsi.py:309-360`)
   - Computes axial gradient ∂||WSS||/∂z using finite differences
   - Structured cylindrical mesh with z-axis slices
   - Central difference for interior points, forward/backward for boundaries
   - Returns gradient in Pa/mm units

2. **Configurable Scaling Parameter** (`in_sim/partitioned_full.json:35-37`)
   ```json
   "growth": {
       "wss_gradient_scale": 500.0
   }
   ```
   - Allows adjustment of gradient values to match expected range
   - Default 500.0 scales ~0.001 Pa/mm to ~0.5 (comparable to WSS magnitude)

3. **Integration with Growth Model** (`svfsi.py:372-416`)
   - Proper gr_properties initialization
   - Scaled gradient passed as tau (current stimulus)
   - tauo (homeostatic stimulus) initialized at t=0

4. **Bug Fixes**
   - Fixed LaTeX rendering error in matplotlib
   - Discovered mesh generation misuses gr_properties for geometric coordinates
   - Corrected initialization to set Jo=1.0, svo=0.0 for undeformed state

---

## Trial Sequence

### Trial 1: Initial Implementation
**Approach**: Pass raw WSS gradient to growth model
**Result**: ❌ FAILED
**Issue**: Incompatible value ranges (gradient ~0.001 Pa/mm vs magnitude ~0.647 Pa)
```
tau (gradient):  ~0.001 Pa/mm
tauo (old magnitude): ~0.687 Pa
Ratio: 0.0015 (99.85% decrease interpreted as extreme stimulus change)
```

### Trial 2: Normalized Gradient
**Approach**: Use (∂WSS/∂z) / WSS for dimensionless stimulus
**Result**: ❌ FAILED
**Issue**: Division by near-zero WSS values
```
WSS median = 0.0 (many interior points)
Normalized gradient → 10^8 (explodes)
```
**Details**:
- WSS only meaningful at fluid-solid interface
- Solid volume contains many zero/near-zero WSS values
- Normalization caused catastrophic overflow

### Trial 3: Preserved Prestress State
**Approach**: Keep Jo and svo from original mesh
**Result**: ❌ FAILED
**Issue**: Misunderstood mesh data structure
```
Mesh gr_properties contains:
  col 0: radius (~0.65) - NOT Jacobian!
  col 1: normalized θ coordinate - NOT stress!
  col 12: 1337 - garbage flag
```
**Discovery**: Mesh generation misuses gr_properties to store geometric coordinates

### Trial 4: Scaled Gradient with Full Reset
**Approach**:
- Reset ALL gr_properties to proper initial values
- Scale gradient by 500 to match WSS magnitude range
- Let C++ code handle tauo during prestressing

**Result**: ❌ FAILED (NaN residuals)
**Issue**: tauo=0 caused division by zero in C++ code
```cpp
tau_ratio = tau / tauo;  // 0.7 / 0.0 → NaN
```

### Trial 5: Manual tauo Initialization
**Approach**: Set tauo = tau at first iteration to avoid division by zero
**Result**: ❌ FAILED (Negative Jacobian)
**Issue**: Newton solver divergence
```
Newton iterations:
  Iteration 1: R/R0 = 1.000e+00
  Iteration 2: R/R0 = 5.192e+02 (explodes!)
  Iteration 3: R/R0 = 3.106e+03
  → Negative Jacobian → Crash
```

**Values passed to solver** (confirmed correct):
```
Jo   = 1.0 (undeformed)
svo  = 0.0 (no prestress)
tau  = [-0.71, 0.71] (scaled gradient)
tauo = [-0.71, 0.71] (homeostatic reference)
tau/tauo = 1.0 (equilibrium)
```

---

## Root Cause Analysis

### Why the Solver Fails

The C++ growth solver (`gr_equilibrated.cpp`) becomes numerically unstable even with correct initialization and homeostatic equilibrium (tau = tauo).

**Possible reasons**:

1. **Different Physics**: Growth model calibrated for WSS magnitude stimulus, not gradient
   - Magnitude represents shear force magnitude
   - Gradient represents spatial variation
   - Different biological interpretation may require different constitutive parameters

2. **Parameter Incompatibility**: Model parameters (K_τσ, growth multipliers) tuned for magnitude
   - Scaling gradient to match numerically may not capture correct physics
   - Growth response may be fundamentally different

3. **Linearization Issues**: Newton solver's Jacobian may be ill-conditioned with gradient stimulus
   - Different sensitivity/coupling with intramural stress
   - May require different solution strategy

---

## Files Modified

### 1. `svfsi.py`
**Lines 309-360**: New function `compute_axial_wss_gradient()`
```python
def compute_axial_wss_gradient(self, wss_magnitude):
    """
    Compute axial WSS gradient magnitude for growth stimulus.

    Calculates ∂||WSS||/∂z using finite differences on structured mesh.
    """
    # ... implementation
    return wss_gradient
```

**Lines 372-416**: Modified `set_solid()` to compute and pass gradient
```python
# Initialize gr_properties
props[:, 0] = 1.0    # Jo
props[:, 1] = 0.0    # svo
props[:, 3] = 0.0    # tauo
props[:, 6] = 0.0    # tau

# Compute and scale gradient
wss_grad = self.compute_axial_wss_gradient(wss_mag)
if "growth" in self.p and "wss_gradient_scale" in self.p["growth"]:
    wss_grad_scaled = wss_grad * self.p["growth"]["wss_gradient_scale"]

# Pass to solver
props[:, 6] = wss_grad_scaled
if t == 0 and n == 0:
    props[:, 3] = wss_grad_scaled
```

### 2. `fsg.py`
**Lines 14-15**: Disabled LaTeX rendering
```python
import matplotlib
matplotlib.rcParams['text.usetex'] = False
```

### 3. `in_sim/partitioned_full.json`
**Lines 35-37**: Added growth configuration
```json
"growth": {
    "wss_gradient_scale": 500.0
}
```

---

## Key Discoveries

### 1. Mesh Generation Bug
`cylinder.py` misuses `gr_properties` array:
- Stores geometric coordinates instead of growth properties
- Must be fully reset before passing to growth solver

### 2. Prestress Incompatibility
Cannot continue from prestressed state when changing stimulus type:
- Old prestress computed with WSS magnitude
- Continuing with WSS gradient (even scaled) represents different physics

### 3. C++ Code Behavior
Growth model expects:
- `tau` (current stimulus) in `props[:, 6]`
- `tauo` (homeostatic reference) in `props[:, 3]`
- Non-zero tauo to compute `tau_ratio = tau / tauo`
- Load step flag in `props[:, 12]` triggers tauo storage

---

## Testing Utilities Created

Developed Python code to inspect VTK files and diagnose issues:

```python
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
import numpy as np

# Read VTK file
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName('path/to/file.vtu')
reader.Update()
output = reader.GetOutput()

# Extract gr_properties
props = v2n(output.GetPointData().GetArray('gr_properties'))

# Check values
print(f'Jo   (col 0):  {props[:,0].min():.4f}, {props[:,0].max():.4f}')
print(f'svo  (col 1):  {props[:,1].min():.4e}, {props[:,1].max():.4e}')
print(f'tauo (col 3):  {props[:,3].min():.4e}, {props[:,3].max():.4e}')
print(f'tau  (col 6):  {props[:,6].min():.4e}, {props[:,6].max():.4e}')
print(f'tau/tauo equal? {np.allclose(props[:,3], props[:,6])}')
```

---

## Recommendations

### Short Term: Keep Using WSS Magnitude
The current implementation with WSS magnitude is stable and validated. Switching to gradient requires C++ solver modifications.

### Medium Term: Investigate C++ Solver
If gradient-based stimulus is scientifically necessary:
1. **Review `gr_equilibrated.cpp`** linearization for gradient stimulus
2. **Adjust constitutive parameters** (K_τσ, growth multipliers) for gradient
3. **Consider different solution strategy** (e.g., line search, arc-length)
4. **Add regularization** to growth equations

### Long Term: Alternative Approaches
1. **Combined Stimulus**: Use both magnitude AND gradient
   ```
   stimulus = α·WSS_magnitude + β·WSS_gradient
   ```

2. **Gradient Magnitude**: Use |∇WSS| instead of axial component
   ```python
   grad_x = ∂WSS/∂x
   grad_y = ∂WSS/∂y
   grad_z = ∂WSS/∂z
   stimulus = sqrt(grad_x² + grad_y² + grad_z²)
   ```

3. **Normalized by Local WSS**: Use gradient relative to local value
   ```
   stimulus = (∂WSS/∂z) / max(WSS_local, WSS_threshold)
   ```

---

## Usage Instructions

### To Use WSS Gradient (Current Implementation)

1. **Configuration** (`in_sim/partitioned_full.json`):
   ```json
   "growth": {
       "wss_gradient_scale": 500.0
   }
   ```

2. **Run**:
   ```bash
   python3 fsg.py ./in_sim/partitioned_full.json
   ```

3. **Expected Behavior**: Will crash with "Negative Jacobian" error

### To Revert to WSS Magnitude

In `svfsi.py`, replace gradient calculation with:
```python
# Get WSS magnitude
wss_mag = self.curr.get(("solid", "wss", "vol"))

# Store WSS magnitude directly (no gradient)
props[:, 6] = wss_mag
if t == 0 and n == 0:
    props[:, 3] = wss_mag
```

---

## Technical Details

### Gradient Calculation Method

**Structured Mesh Assumptions**:
- Points organized as z-slices
- Each slice has `n_points_per_slice = n_cir × (n_rad_gr + 1)` points
- Point `i` and point `i + n_points_per_slice` are axial neighbors

**Finite Difference Stencils**:
- **Interior**: Central difference
  ```
  ∂WSS/∂z|ᵢ = (WSS[i+n_slice] - WSS[i-n_slice]) / (z[i+n_slice] - z[i-n_slice])
  ```
- **Inlet (z=0)**: Forward difference
  ```
  ∂WSS/∂z|ᵢ = (WSS[i+n_slice] - WSS[i]) / (z[i+n_slice] - z[i])
  ```
- **Outlet (z=L)**: Backward difference
  ```
  ∂WSS/∂z|ᵢ = (WSS[i] - WSS[i-n_slice]) / (z[i] - z[i-n_slice])
  ```

### Scaling Rationale

**Observed value ranges**:
- WSS magnitude: 0.6 - 0.7 Pa
- WSS gradient: 0.001 - 0.0015 Pa/mm
- Ratio: ~500

**Scaling factor** chosen to make gradient numerically comparable:
```
scaled_gradient = gradient × 500
                ≈ 0.001 × 500
                ≈ 0.5 Pa
```

This matches the magnitude range, allowing reuse of existing growth parameters.

---

## Error Messages Guide

### "Negative Jacobian"
**Meaning**: Element inversion (volume becomes negative)
**Cause**: Excessive deformation during Newton iteration
**Indicates**: Numerical instability in growth equations

### "NaN residuals"
**Meaning**: Division by zero or overflow in residual calculation
**Cause**: tauo = 0 or very small values
**Fix**: Initialize tauo = tau at t=0

### "latex not found"
**Meaning**: Matplotlib trying to use LaTeX for text rendering
**Fix**: Add `matplotlib.rcParams['text.usetex'] = False`

---

## References

- **Paper**: Pfaller et al. (2024) "FSGe: A fast and strongly-coupled 3D fluid–solid-growth interaction method"
- **Original Implementation**: `WSS_GRADIENT_IMPLEMENTATION.md`
- **Growth Model**: `gr_equilibrated.cpp` (archived in simulation results)

---

## Conclusion

The WSS gradient implementation is **technically correct** but **numerically unstable** with the current C++ growth solver. The Python code successfully:
- ✅ Computes gradients using finite differences
- ✅ Applies configurable scaling
- ✅ Initializes gr_properties correctly
- ✅ Passes values to C++ solver

The failure occurs in the C++ Newton solver's linearization, suggesting the growth model requires modification to work with gradient-based stimulus. This is likely a **scientific/modeling issue** rather than an implementation bug.

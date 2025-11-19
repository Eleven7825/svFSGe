# FSGe VTK Testing and Debugging Utilities

**Purpose**: Reusable Python code for inspecting VTK files and debugging FSGe simulations
**Usage**: Copy and adapt code snippets for specific debugging needs

---

## Quick Reference

### Common Debugging Tasks

1. **Check gr_properties values** → [Section 1](#1-inspect-gr_properties-array)
2. **Verify WSS values** → [Section 2](#2-check-wss-values)
3. **Compare before/after files** → [Section 3](#3-compare-vtk-files)
4. **Check mesh structure** → [Section 4](#4-verify-mesh-structure)
5. **Find latest simulation** → [Section 5](#5-find-latest-simulation-results)

---

## 1. Inspect gr_properties Array

### Basic Inspection

```python
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
import numpy as np

def inspect_gr_properties(filename):
    """
    Read and display gr_properties from a VTK file.

    Args:
        filename: Path to .vtu or .vtp file
    """
    # Read file
    if filename.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()

    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()

    # Extract gr_properties
    if not output.GetPointData().HasArray('gr_properties'):
        print(f"ERROR: No 'gr_properties' array in {filename}")
        return None

    props = v2n(output.GetPointData().GetArray('gr_properties'))

    # Display key columns
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print(f"gr_properties shape: {props.shape}")
    print(f"{'='*60}\n")

    columns = {
        0: 'Jo (Jacobian)',
        1: 'svo (stress)',
        3: 'tauo (homeostatic stimulus)',
        6: 'tau (current stimulus)',
        7: 'time',
        12: 'load_step_flag'
    }

    for col, name in columns.items():
        if col < props.shape[1]:
            vals = props[:, col]
            print(f"{name:25s} (col {col:2d}):")
            print(f"  min={vals.min():12.6e}  max={vals.max():12.6e}")
            print(f"  mean={vals.mean():12.6e}  std={vals.std():12.6e}")

            # Check for special values
            n_nan = np.sum(np.isnan(vals))
            n_inf = np.sum(np.isinf(vals))
            n_zero = np.sum(vals == 0)
            if n_nan > 0 or n_inf > 0:
                print(f"  WARNING: {n_nan} NaN, {n_inf} Inf values")
            if n_zero > 0:
                print(f"  {n_zero} zero values ({100*n_zero/len(vals):.1f}%)")

    # Check tau/tauo ratio
    if 3 < props.shape[1] and 6 < props.shape[1]:
        print(f"\n{'Ratio Analysis':25s}")
        mask = (np.abs(props[:, 3]) > 1e-12) & (np.abs(props[:, 6]) > 1e-12)
        if np.any(mask):
            ratio = props[mask, 6] / props[mask, 3]
            print(f"  tau/tauo (non-zero): min={ratio.min():.6f}, max={ratio.max():.6f}")
            print(f"  mean={ratio.mean():.6f}, near 1.0? {np.allclose(ratio, 1.0, rtol=0.01)}")
        else:
            print(f"  All tau or tauo values are zero")

    print(f"\n{'='*60}\n")

    return props

# Example usage
props = inspect_gr_properties('partitioned_2025-XX-XX/solid.vtu')
```

### Quick Check (One-liner)

```python
# Quick check of specific file
import vtk; from vtk.util.numpy_support import vtk_to_numpy as v2n; import numpy as np; \
reader = vtk.vtkXMLUnstructuredGridReader(); reader.SetFileName('FILE.vtu'); reader.Update(); \
props = v2n(reader.GetOutput().GetPointData().GetArray('gr_properties')); \
print(f'Jo: {props[:,0].min():.3f}-{props[:,0].max():.3f}, tau: {props[:,6].min():.3e}-{props[:,6].max():.3e}')
```

---

## 2. Check WSS Values

### Inspect WSS from Fluid Solution

```python
def check_wss_values(filename):
    """
    Check WSS vector and magnitude from fluid solution.

    Args:
        filename: Path to fluid output .vtu file
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()

    # Check for WSS array
    if not output.GetPointData().HasArray('WSS'):
        print(f"ERROR: No 'WSS' array in {filename}")
        print("Available arrays:")
        for i in range(output.GetPointData().GetNumberOfArrays()):
            print(f"  - {output.GetPointData().GetArrayName(i)}")
        return

    # Get WSS vector
    wss_vec = v2n(output.GetPointData().GetArray('WSS'))
    wss_mag = np.linalg.norm(wss_vec, axis=1)

    print(f"\n{'='*60}")
    print(f"WSS Analysis: {filename}")
    print(f"{'='*60}\n")

    print(f"Number of points: {len(wss_mag)}")
    print(f"\nWSS Magnitude:")
    print(f"  min:    {wss_mag.min():.6e}")
    print(f"  max:    {wss_mag.max():.6e}")
    print(f"  mean:   {wss_mag.mean():.6e}")
    print(f"  median: {np.median(wss_mag):.6e}")
    print(f"  std:    {wss_mag.std():.6e}")

    # Check distribution
    n_zero = np.sum(wss_mag < 1e-10)
    n_small = np.sum(wss_mag < 1e-6)
    print(f"\nDistribution:")
    print(f"  Near zero (<1e-10): {n_zero} ({100*n_zero/len(wss_mag):.1f}%)")
    print(f"  Small (<1e-6):      {n_small} ({100*n_small/len(wss_mag):.1f}%)")

    # WSS vector components
    print(f"\nWSS Vector Components:")
    for i, comp in enumerate(['x', 'y', 'z']):
        print(f"  {comp}: min={wss_vec[:,i].min():.6e}, max={wss_vec[:,i].max():.6e}")

    print(f"\n{'='*60}\n")

    return wss_vec, wss_mag

# Example usage
wss_vec, wss_mag = check_wss_values('partitioned_2025-XX-XX/steady/steady_001.vtu')
```

### Test Gradient Calculation

```python
def test_gradient_calculation(wss_mag, points, n_points_per_slice):
    """
    Test the WSS gradient calculation independently.

    Args:
        wss_mag: Array of WSS magnitudes
        points: Array of point coordinates (N, 3)
        n_points_per_slice: Number of points per z-slice
    """
    n_points = len(points)
    wss_gradient = np.zeros(n_points)

    print(f"Testing gradient calculation:")
    print(f"  Total points: {n_points}")
    print(f"  Points per slice: {n_points_per_slice}")
    print(f"  Number of slices: {n_points // n_points_per_slice}")

    for i in range(n_points):
        idx_prev = i - n_points_per_slice
        idx_next = i + n_points_per_slice

        if idx_prev >= 0 and idx_next < n_points:
            # Interior: central difference
            dz = points[idx_next, 2] - points[idx_prev, 2]
            dwss = wss_mag[idx_next] - wss_mag[idx_prev]
            wss_gradient[i] = dwss / dz if dz > 1e-12 else 0.0
        elif idx_prev < 0:
            # Forward difference
            dz = points[idx_next, 2] - points[i, 2]
            dwss = wss_mag[idx_next] - wss_mag[i]
            wss_gradient[i] = dwss / dz if dz > 1e-12 else 0.0
        else:
            # Backward difference
            dz = points[i, 2] - points[idx_prev, 2]
            dwss = wss_mag[i] - wss_mag[idx_prev]
            wss_gradient[i] = dwss / dz if dz > 1e-12 else 0.0

    print(f"\nGradient Results:")
    print(f"  min:  {wss_gradient.min():.6e}")
    print(f"  max:  {wss_gradient.max():.6e}")
    print(f"  mean: {wss_gradient.mean():.6e}")
    print(f"  std:  {wss_gradient.std():.6e}")

    return wss_gradient
```

---

## 3. Compare VTK Files

### Compare Two gr_properties Arrays

```python
def compare_gr_properties(file1, file2, label1="File 1", label2="File 2"):
    """
    Compare gr_properties between two VTK files.

    Args:
        file1, file2: Paths to VTK files
        label1, label2: Labels for display
    """
    def read_props(filename):
        if filename.endswith('.vtp'):
            reader = vtk.vtkXMLPolyDataReader()
        else:
            reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        return v2n(reader.GetOutput().GetPointData().GetArray('gr_properties'))

    props1 = read_props(file1)
    props2 = read_props(file2)

    print(f"\n{'='*70}")
    print(f"Comparing gr_properties")
    print(f"{'='*70}\n")
    print(f"{label1:30s}: {file1}")
    print(f"{label2:30s}: {file2}\n")

    columns = {0: 'Jo', 1: 'svo', 3: 'tauo', 6: 'tau', 7: 'time', 12: 'flag'}

    print(f"{'Column':10s} {'Name':15s} {label1:>15s} {label2:>15s} {'Difference':>15s}")
    print(f"{'-'*70}")

    for col, name in columns.items():
        if col < min(props1.shape[1], props2.shape[1]):
            mean1 = props1[:, col].mean()
            mean2 = props2[:, col].mean()
            diff = mean2 - mean1
            print(f"{col:<10d} {name:15s} {mean1:15.6e} {mean2:15.6e} {diff:15.6e}")

    # Check if arrays are close
    if props1.shape == props2.shape:
        print(f"\n{'Arrays identical?':30s} {np.allclose(props1, props2)}")
        max_diff = np.abs(props1 - props2).max()
        print(f"{'Maximum difference:':30s} {max_diff:.6e}")
    else:
        print(f"\nWARNING: Different shapes: {props1.shape} vs {props2.shape}")

    print(f"\n{'='*70}\n")

# Example usage
compare_gr_properties(
    'partitioned_A/solid.vtu',
    'partitioned_B/solid.vtu',
    label1="Before fix",
    label2="After fix"
)
```

---

## 4. Verify Mesh Structure

### Check Mesh Parameters

```python
def verify_mesh_structure(filename, expected_n_cir, expected_n_rad_gr):
    """
    Verify the structured mesh has expected parameters.

    Args:
        filename: Path to mesh file
        expected_n_cir: Expected circumferential points
        expected_n_rad_gr: Expected radial layers
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()

    points = v2n(output.GetPoints().GetData())
    n_points = len(points)

    # Calculate expected values
    n_points_per_slice = expected_n_cir * (expected_n_rad_gr + 1)
    n_slices = n_points // n_points_per_slice

    print(f"\n{'='*60}")
    print(f"Mesh Structure Verification")
    print(f"{'='*60}\n")
    print(f"File: {filename}")
    print(f"\nExpected:")
    print(f"  n_cir:              {expected_n_cir}")
    print(f"  n_rad_gr:           {expected_n_rad_gr}")
    print(f"  Points per slice:   {n_points_per_slice}")
    print(f"\nActual:")
    print(f"  Total points:       {n_points}")
    print(f"  Calculated slices:  {n_slices}")
    print(f"  Remainder:          {n_points % n_points_per_slice}")

    # Check z-coordinates
    z_coords = points[:, 2]
    print(f"\nZ-axis:")
    print(f"  min: {z_coords.min():.6f}")
    print(f"  max: {z_coords.max():.6f}")
    print(f"  range: {z_coords.max() - z_coords.min():.6f}")

    # Check if structured
    is_structured = (n_points % n_points_per_slice) == 0
    print(f"\nStructured mesh: {is_structured}")

    if is_structured:
        # Sample check: verify neighbor relationship
        test_idx = n_points_per_slice  # Second slice, first point
        neighbor_idx = test_idx + n_points_per_slice  # Third slice, first point

        if neighbor_idx < n_points:
            dz = points[neighbor_idx, 2] - points[test_idx, 2]
            dr = np.linalg.norm(points[neighbor_idx, :2] - points[test_idx, :2])
            print(f"\nNeighbor check (point {test_idx} → {neighbor_idx}):")
            print(f"  Δz (should be >0): {dz:.6f}")
            print(f"  Δr (should be ~0): {dr:.6e}")
            print(f"  Valid axial neighbors: {dz > 1e-6 and dr < 1e-6}")

    print(f"\n{'='*60}\n")

    return is_structured

# Example usage
verify_mesh_structure(
    'partitioned_2025-XX-XX/mesh_tube_fsi/solid/mesh-complete.mesh.vtu',
    expected_n_cir=32,
    expected_n_rad_gr=1
)
```

---

## 5. Find Latest Simulation Results

### Bash Helper Functions

```bash
# Find latest simulation directory
latest_sim() {
    ls -td partitioned_2025-* 2>/dev/null | head -1
}

# Find latest solid log
latest_solid_log() {
    ls -t partitioned_*/partitioned/solid_*.log 2>/dev/null | head -1
}

# Check if simulation succeeded
check_sim_status() {
    local latest=$(latest_sim)
    if [ -z "$latest" ]; then
        echo "No simulations found"
        return 1
    fi

    echo "Latest: $latest"

    # Check for error in solid log
    local log=$(ls -t $latest/partitioned/solid_*.log 2>/dev/null | head -1)
    if [ -f "$log" ]; then
        if grep -q "Negative Jacobian\|NaN\|ERROR" "$log"; then
            echo "Status: FAILED (check $log)"
            return 1
        else
            echo "Status: May have succeeded"
            return 0
        fi
    fi
}

# Usage:
# check_sim_status
```

### Python Helper Functions

```python
import glob
import os
from datetime import datetime

def find_latest_simulation():
    """Find the most recent simulation directory."""
    dirs = glob.glob('partitioned_2025-*/')
    if not dirs:
        return None

    # Sort by modification time
    latest = max(dirs, key=os.path.getmtime)
    return latest.rstrip('/')

def get_simulation_files(sim_dir=None):
    """
    Get paths to key simulation files.

    Returns:
        dict: Paths to important files
    """
    if sim_dir is None:
        sim_dir = find_latest_simulation()
        if sim_dir is None:
            print("No simulation directories found")
            return None

    files = {
        'solid_input': f'{sim_dir}/solid.vtu',
        'solid_mesh': f'{sim_dir}/mesh_tube_fsi/solid/mesh-complete.mesh.vtu',
        'fluid_steady': f'{sim_dir}/steady/steady_001.vtu',
        'latest_solid_out': None,
        'latest_solid_log': None,
    }

    # Find latest outputs in partitioned directory
    solid_outs = glob.glob(f'{sim_dir}/partitioned/solid_out_*.vtu')
    if solid_outs:
        files['latest_solid_out'] = max(solid_outs, key=os.path.getmtime)

    solid_logs = glob.glob(f'{sim_dir}/partitioned/solid_*.log')
    if solid_logs:
        files['latest_solid_log'] = max(solid_logs, key=os.path.getmtime)

    return files

def quick_diagnosis():
    """Quick diagnosis of latest simulation."""
    sim_dir = find_latest_simulation()
    if not sim_dir:
        print("No simulations found")
        return

    print(f"\n{'='*60}")
    print(f"Quick Diagnosis: {sim_dir}")
    print(f"{'='*60}\n")

    files = get_simulation_files(sim_dir)

    # Check solid input
    if os.path.exists(files['solid_input']):
        print("✓ Solid input exists")
        props = inspect_gr_properties(files['solid_input'])
    else:
        print("✗ Solid input missing")

    # Check log
    if files['latest_solid_log'] and os.path.exists(files['latest_solid_log']):
        with open(files['latest_solid_log'], 'r') as f:
            log_content = f.read()

        if 'Negative Jacobian' in log_content:
            print("✗ Simulation failed: Negative Jacobian")
        elif 'nan' in log_content.lower():
            print("✗ Simulation failed: NaN values")
        else:
            print("? Simulation may have completed")

    print(f"\n{'='*60}\n")

# Example usage
quick_diagnosis()
```

---

## 6. Complete Debugging Workflow

### Step-by-Step Debugging Process

```python
def debug_simulation():
    """Complete debugging workflow for FSGe simulation."""

    print("\n" + "="*70)
    print("FSGe Simulation Debugging Workflow")
    print("="*70 + "\n")

    # Step 1: Find latest simulation
    print("Step 1: Locating latest simulation...")
    sim_dir = find_latest_simulation()
    if not sim_dir:
        print("ERROR: No simulation directories found")
        return
    print(f"Found: {sim_dir}\n")

    # Step 2: Get file paths
    print("Step 2: Checking for required files...")
    files = get_simulation_files(sim_dir)

    for name, path in files.items():
        if path and os.path.exists(path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (missing)")
    print()

    # Step 3: Check solid input gr_properties
    print("Step 3: Inspecting solid input gr_properties...")
    if os.path.exists(files['solid_input']):
        props = inspect_gr_properties(files['solid_input'])

        # Validate values
        issues = []
        if props is not None:
            if not np.allclose(props[:, 0], 1.0, rtol=0.1):
                issues.append("Jo should be ~1.0 (undeformed)")
            if not np.allclose(props[:, 1], 0.0, atol=0.01):
                issues.append("svo should be ~0.0 (no prestress)")
            if np.any(props[:, 3] == 0) and np.any(props[:, 6] != 0):
                issues.append("tauo is 0 but tau is not (division by zero!)")

            if issues:
                print("\n  ISSUES DETECTED:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print("\n  Values look reasonable")
    print()

    # Step 4: Check solver log
    print("Step 4: Checking solver log...")
    if files['latest_solid_log'] and os.path.exists(files['latest_solid_log']):
        with open(files['latest_solid_log'], 'r') as f:
            lines = f.readlines()

        # Check last few lines
        print("  Last 10 lines of log:")
        for line in lines[-10:]:
            print(f"    {line.rstrip()}")

        # Check for errors
        log_content = ''.join(lines)
        if 'Negative Jacobian' in log_content:
            print("\n  ERROR: Negative Jacobian detected")
            print("  → Newton solver diverged, element inversion occurred")
        elif 'nan' in log_content.lower():
            print("\n  ERROR: NaN values detected")
            print("  → Likely division by zero (check tauo values)")
        else:
            print("\n  No obvious errors in log")
    print()

    # Step 5: Recommendations
    print("Step 5: Recommendations...")
    print("  See WSS_GRADIENT_TRIALS.md for detailed troubleshooting")
    print("  Common fixes:")
    print("    - Ensure tauo is initialized (non-zero)")
    print("    - Check scaling factor is appropriate")
    print("    - Verify Jo=1.0, svo=0.0 for undeformed state")
    print()

    print("="*70 + "\n")

# Run complete debugging
debug_simulation()
```

---

## 7. Utility Collection

### All-in-One Script

Save this as `debug_fsge.py`:

```python
#!/usr/bin/env python3
"""
FSGe Debugging Utilities

Usage:
    python debug_fsge.py                    # Full diagnosis
    python debug_fsge.py --file FILE.vtu    # Inspect specific file
    python debug_fsge.py --compare F1 F2    # Compare two files
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
import numpy as np
import glob
import os
import sys

# [Include all functions from above sections]
# - inspect_gr_properties()
# - check_wss_values()
# - compare_gr_properties()
# - verify_mesh_structure()
# - find_latest_simulation()
# - get_simulation_files()
# - debug_simulation()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FSGe Debugging Utilities')
    parser.add_argument('--file', help='Inspect specific VTK file')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare two VTK files')
    parser.add_argument('--mesh', help='Verify mesh structure')
    parser.add_argument('--wss', help='Check WSS values in file')

    args = parser.parse_args()

    if args.file:
        inspect_gr_properties(args.file)
    elif args.compare:
        compare_gr_properties(args.compare[0], args.compare[1])
    elif args.mesh:
        verify_mesh_structure(args.mesh, n_cir=32, n_rad_gr=1)
    elif args.wss:
        check_wss_values(args.wss)
    else:
        # Default: full debugging
        debug_simulation()
```

---

## Quick Reference Commands

### Most Common Debugging Tasks

```bash
# 1. Find and inspect latest simulation
python3 -c "
import sys; sys.path.append('.')
from debug_fsge import *
debug_simulation()
"

# 2. Check specific file
python3 -c "
from debug_fsge import inspect_gr_properties
inspect_gr_properties('partitioned_XXXX/solid.vtu')
"

# 3. Compare before/after
python3 debug_fsge.py --compare before/solid.vtu after/solid.vtu

# 4. Check solver log
tail -50 $(ls -t partitioned_*/partitioned/solid_*.log | head -1)

# 5. Verify values are in expected range
python3 -c "
import vtk; from vtk.util.numpy_support import vtk_to_numpy as v2n
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName('FILE.vtu')
reader.Update()
props = v2n(reader.GetOutput().GetPointData().GetArray('gr_properties'))
print(f'Jo={props[:,0].mean():.2f} (expect 1.0)')
print(f'svo={props[:,1].mean():.2e} (expect ~0)')
print(f'tau={props[:,6].mean():.2e}, tauo={props[:,3].mean():.2e}')
print(f'Ratio={props[:,6].mean()/props[:,3].mean():.2f} (expect ~1.0)')
"
```

---

## Troubleshooting Guide

### Common Issues and Solutions

| Issue | Symptoms | Check | Solution |
|-------|----------|-------|----------|
| **Division by zero** | NaN residuals in solver | `tauo == 0` but `tau != 0` | Initialize `tauo = tau` at t=0 |
| **Negative Jacobian** | Solver crash after 1-3 iterations | Residual explodes (1.0 → 500+) | Check growth parameters, initial conditions |
| **Wrong value range** | Values don't match expected | tau ~0.001 vs expected ~0.6 | Apply scaling factor |
| **Bad initialization** | Jo != 1.0, svo != 0 | Mesh generation sets wrong values | Reset gr_properties in `set_solid()` |
| **Corrupted mesh** | Geometric coordinates in gr_props | col 0 = radius, col 12 = 1337 | Re-initialize all columns |

### Expected Value Ranges

```
Jo (col 0):     ~1.0          (undeformed: exactly 1.0)
svo (col 1):    ~0.0          (no prestress: exactly 0.0)
                or ~0.5-1.0   (with prestress)
tauo (col 3):   ~0.001        (gradient, unscaled)
                or ~0.5-0.7   (magnitude or scaled gradient)
tau (col 6):    Same as tauo at t=0
time (col 7):   1, 2, 3, ...  (timestep + 1)
flag (col 12):  0 or 1        (NOT 1337!)
```

---

## Additional Resources

- **Main documentation**: `WSS_GRADIENT_IMPLEMENTATION.md`
- **Trial summary**: `WSS_GRADIENT_TRIALS.md`
- **VTK documentation**: https://vtk.org/doc/nightly/html/
- **NumPy integration**: `vtk.util.numpy_support`

---

## Notes for Future Sessions

1. **Always check gr_properties** before and after modifications
2. **Verify tau/tauo ratio** at homeostatic state (should be ~1.0)
3. **Check solver logs** for NaN or Negative Jacobian errors
4. **Use scaling factor** when switching between stimulus types
5. **Reset mesh-generated values** (they contain geometric coordinates, not growth properties)

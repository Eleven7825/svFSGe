# Post-Processing Fluid-Only Simulation Results

## Overview

The `post_fluid.py` script extracts and analyzes results from your fluid-only svFSI simulation.

## What it extracts:

✅ **Pressure** (in mmHg)
✅ **Velocity** (in mm/s)
✅ **Wall Shear Stress (WSS)** (in dyne/cm²)
✅ **Traction** (if available, in Pa)

## Quick Start

### Basic usage (summary only):
```bash
./run_post.sh steady --no-plot
```

### Export to CSV:
```bash
./run_post.sh steady --csv --no-plot
```

### Generate time series plots:
```bash
./run_post.sh steady
```

### Custom file pattern:
```bash
./run_post.sh steady --pattern "steady_*.vtu"
```

### Custom output directory:
```bash
./run_post.sh steady --output my_results --csv
```

## Command-line options:

- `--no-plot`: Skip generating plots (faster)
- `--csv`: Export results to CSV file
- `-o, --output DIR`: Specify output directory (default: ./post_results)
- `-p, --pattern`: File pattern for VTU files (default: steady_*.vtu)

## Output files:

1. **fluid_results.csv**: Time series of mean/max/min values for all quantities
2. **fluid_time_series.pdf**: Plots showing evolution over time

## Example CSV output:

```
timestep,pressure_mean,pressure_max,pressure_min,velocity_mean,velocity_max,wss_mean,wss_max
0,21.869,22.736,20.981,382.592,694.789,0.001263,0.012841
1,16.022,19.633,10.402,1655.239,3585.879,0.006083,0.087101
...
```

## Notes:

- The script automatically reads all `steady_*.vtu` files from the specified directory
- WSS values are extracted if present in the VTU files
- Memory warnings at the end can be ignored - processing completes successfully
- Results are saved in `./post_results/` by default

## Comparison with original post.py:

The original `../post.py` script:
- Was designed for FSGe (fluid-structure-growth) simulations
- Expects files named `tube_*.vtu` or `gr_*.vtu`
- Does NOT extract WSS for fluid domain

This modified script:
- Works with your `steady_*.vtu` files
- Extracts WSS and other fluid quantities
- Provides simpler interface for fluid-only analysis

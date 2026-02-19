#!/usr/bin/env python
# coding=utf-8
"""
Modified post.py for fluid-only simulations
Extracts pressure, velocity, and wall shear stress from svFSI results
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk_functions import read_geo

plt.rcParams.update({"text.usetex": False})

# Field labels (kg-mm-s unit system)
f_labels = {
    "pressure": ["Pressure [kg/(mm·s²)]"],
    "velocity": ["Velocity [mm/s]"],
    "wss": ["Wall Shear Stress [kg/(mm·s²)]"],
    "traction": ["Traction [kg/(mm·s²)]"],
}


def extract_results_fluid(res, wall_face_name="interface"):
    """
    Extract fluid results including WSS from wall

    Args:
        res: VTU geometry object
        wall_face_name: Name of wall face (default: "interface")

    Returns:
        dict with pressure, velocity, wss data
    """
    results = {}

    # Extract volume data
    point_data = res.GetPointData()

    # Pressure
    if point_data.HasArray("Pressure"):
        pressure = v2n(point_data.GetArray("Pressure"))
        results["pressure"] = pressure
        results["pressure_mean"] = np.mean(pressure)
        results["pressure_max"] = np.max(pressure)
        results["pressure_min"] = np.min(pressure)

    # Velocity
    if point_data.HasArray("Velocity"):
        velocity = v2n(point_data.GetArray("Velocity"))
        velocity_mag = np.linalg.norm(velocity, axis=1)
        results["velocity"] = velocity
        results["velocity_mag"] = velocity_mag
        results["velocity_mean"] = np.mean(velocity_mag)
        results["velocity_max"] = np.max(velocity_mag)

    # Wall Shear Stress - check if it exists
    if point_data.HasArray("WSS"):
        wss = v2n(point_data.GetArray("WSS"))
        if len(wss.shape) > 1:
            wss_mag = np.linalg.norm(wss, axis=1)
        else:
            wss_mag = wss
        results["wss"] = wss
        results["wss_mag"] = wss_mag
        results["wss_mean"] = np.mean(wss_mag)
        results["wss_max"] = np.max(wss_mag)

        # WSS at middle cross-section (filter by z position and non-zero WSS)
        points = v2n(res.GetPoints().GetData())
        z_coords = points[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_mid = (z_min + z_max) / 2
        z_tol = (z_max - z_min) * 0.05  # 5% band around middle
        mid_mask = np.abs(z_coords - z_mid) < z_tol
        wss_mid = wss_mag[mid_mask]
        wall_mask = wss_mid > 1e-10  # exclude interior nodes (zero WSS)
        if np.any(wall_mask):
            results["wss_mid_mean"] = np.mean(wss_mid[wall_mask])
        else:
            results["wss_mid_mean"] = np.mean(wss_mid)

    # Traction - alternative to WSS
    if point_data.HasArray("Traction"):
        traction = v2n(point_data.GetArray("Traction"))
        if len(traction.shape) > 1:
            traction_mag = np.linalg.norm(traction, axis=1)
        else:
            traction_mag = traction
        results["traction"] = traction
        results["traction_mag"] = traction_mag
        results["traction_mean"] = np.mean(traction_mag)
        results["traction_max"] = np.max(traction_mag)

    return results


def read_all_results(output_dir, file_pattern="steady_*.vtu"):
    """
    Read all VTU files from output directory

    Args:
        output_dir: Directory containing VTU files
        file_pattern: Glob pattern for VTU files

    Returns:
        list of results for each timestep
    """
    fname = os.path.join(output_dir, file_pattern)
    files = sorted(glob.glob(fname))

    if not files:
        raise RuntimeError(f"No results found matching {fname}")

    print(f"Found {len(files)} result files")

    results = []
    for fn in files:
        print(f"Reading {os.path.basename(fn)}")
        geo = read_geo(fn).GetOutput()
        res = extract_results_fluid(geo)
        results.append(res)

    return results


def summarize_results(results):
    """
    Print summary statistics for all timesteps
    """
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)

    # Check what fields are available
    fields = results[0].keys()

    if "pressure_mean" in fields:
        pressures = [r["pressure_mean"] for r in results]
        print(f"\nPressure [kg/(mm·s²)]:")
        print(f"  Mean: {np.mean(pressures):.3f}")
        print(f"  Min:  {np.min([r['pressure_min'] for r in results]):.3f}")
        print(f"  Max:  {np.max([r['pressure_max'] for r in results]):.3f}")

    if "velocity_mean" in fields:
        velocities = [r["velocity_mean"] for r in results]
        print(f"\nVelocity [mm/s]:")
        print(f"  Mean: {np.mean(velocities):.3f}")
        print(f"  Max:  {np.max([r['velocity_max'] for r in results]):.3f}")

    if "wss_mean" in fields:
        wss_vals = [r["wss_mean"] for r in results]
        print(f"\nWall Shear Stress [kg/(mm·s²)]:")
        print(f"  Mean: {np.mean(wss_vals):.6f} ({np.mean(wss_vals)*10000:.2f} dyne/cm²)")
        print(f"  Max:  {np.max([r['wss_max'] for r in results]):.6f} ({np.max([r['wss_max'] for r in results])*10000:.2f} dyne/cm²)")

    if "traction_mean" in fields:
        traction_vals = [r["traction_mean"] for r in results]
        print(f"\nTraction [kg/(mm·s²)]:")
        print(f"  Mean: {np.mean(traction_vals):.3f}")
        print(f"  Max:  {np.max([r['traction_max'] for r in results]):.3f}")

    print("\n" + "="*60)


def plot_time_series(results, output_dir):
    """
    Plot time series of fluid quantities

    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    n_steps = len(results)
    time_steps = np.arange(n_steps)

    # Determine which quantities to plot
    plot_quantities = []
    if "pressure_mean" in results[0]:
        plot_quantities.append(("pressure_mean", "Mean Pressure [kg/(mm·s²)]"))
    if "velocity_mean" in results[0]:
        plot_quantities.append(("velocity_mean", "Mean Velocity [mm/s]"))
    if "wss_mean" in results[0]:
        plot_quantities.append(("wss_mean", "Mean WSS [kg/(mm·s²)]"))
    if "traction_mean" in results[0]:
        plot_quantities.append(("traction_mean", "Mean Traction [kg/(mm·s²)]"))

    # Create subplots
    n_plots = len(plot_quantities)
    if n_plots == 0:
        print("No data to plot")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots), dpi=150)
    if n_plots == 1:
        axes = [axes]

    for ax, (field, label) in zip(axes, plot_quantities):
        data = [r[field] for r in results]
        ax.plot(time_steps, data, 'b-', linewidth=2)
        ax.set_xlabel("Time Step")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "fluid_time_series.pdf")
    fig.savefig(fname, bbox_inches="tight")
    print(f"\nSaved time series plot to {fname}")
    plt.close()


def plot_wss_heartbeats(results, output_dir, steps_per_beat):
    """
    Overlay WSS at middle cross-section for each heart beat in one plot.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save the plot
        steps_per_beat: Number of time steps per heart beat
    """
    os.makedirs(output_dir, exist_ok=True)

    if "wss_mid_mean" not in results[0]:
        print("No mid-section WSS data available; skipping heartbeat overlay plot")
        return

    wss_mid = np.array([r["wss_mid_mean"] for r in results])
    n_beats = len(wss_mid) // steps_per_beat

    if n_beats == 0:
        print("Not enough steps for one full beat; skipping heartbeat overlay plot")
        return

    colors = plt.cm.tab10.colors
    beat_steps = np.arange(steps_per_beat)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    for i in range(n_beats):
        start = i * steps_per_beat
        end = start + steps_per_beat
        beat_wss = wss_mid[start:end]
        ax.plot(beat_steps, beat_wss, color=colors[i % len(colors)],
                linewidth=2, label=f"Beat {i + 1}")

    ax.set_xlabel("Step within Beat")
    ax.set_ylabel("Mean WSS at Middle [kg/(mm·s²)]")
    ax.set_title("WSS at Middle Cross-section — Heart Beat Overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "wss_heartbeats.pdf")
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved WSS heartbeat overlay plot to {fname}")
    plt.close()


def export_to_csv(results, output_file):
    """
    Export results to CSV file

    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Determine which fields to export (summary statistics)
    fields = []
    for key in results[0].keys():
        if "_mean" in key or "_max" in key or "_min" in key:
            fields.append(key)

    # Write CSV header
    with open(output_file, 'w') as f:
        f.write("timestep," + ",".join(fields) + "\n")

        # Write data
        for i, res in enumerate(results):
            values = [str(res.get(field, "")) for field in fields]
            f.write(f"{i}," + ",".join(values) + "\n")

    print(f"Exported results to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process fluid-only svFSI simulation results"
    )
    parser.add_argument(
        "output_dir",
        help="Directory containing VTU output files"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="steady_*.vtu",
        help="File pattern for VTU files (default: steady_*.vtu)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for plots and CSV (default: <output_dir>/post)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV"
    )
    parser.add_argument(
        "--steps-per-beat",
        type=int,
        default=None,
        help="Number of time steps per heart beat for the beat overlay plot"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output is None:
        # Use current directory for output if output_dir is not writable
        args.output = os.path.join(".", "post_results")

    # Read all results
    results = read_all_results(args.output_dir, args.pattern)

    # Print summary
    summarize_results(results)

    # Create plots
    if not args.no_plot:
        plot_time_series(results, args.output)
        if args.steps_per_beat is not None:
            plot_wss_heartbeats(results, args.output, args.steps_per_beat)

    # Export to CSV
    if args.csv:
        csv_file = os.path.join(args.output, "fluid_results.csv")
        export_to_csv(results, csv_file)

    print(f"\nPost-processing complete!")


if __name__ == "__main__":
    main()

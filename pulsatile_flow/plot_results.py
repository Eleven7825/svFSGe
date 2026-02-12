#!/usr/bin/env python3
"""
Plot fluid simulation results from CSV file
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_fluid_results(csv_file, output_dir="post_results"):
    """
    Create comprehensive plots from fluid results CSV

    Args:
        csv_file: Path to CSV file
        output_dir: Directory to save plots
    """
    # Read CSV data
    print(f"Reading {csv_file}...")
    data = np.genfromtxt(csv_file, delimiter=',', names=True)

    timesteps = data['timestep']
    n_steps = len(timesteps)

    print(f"Loaded {n_steps} timesteps")

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=150)

    # Plot 1: Pressure
    ax1 = axes[0]
    ax1.plot(timesteps, data['pressure_mean'], 'b-', linewidth=2, label='Mean')
    ax1.fill_between(timesteps, data['pressure_min'], data['pressure_max'],
                      alpha=0.3, color='blue', label='Min-Max Range')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Pressure [kg/(mm·s²)]', fontsize=12)
    ax1.set_title('Pressure Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Plot 2: Velocity
    ax2 = axes[1]
    ax2.plot(timesteps, data['velocity_mean'], 'r-', linewidth=2, label='Mean')
    ax2.plot(timesteps, data['velocity_max'], 'r--', linewidth=1.5, alpha=0.7, label='Max')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Velocity [mm/s]', fontsize=12)
    ax2.set_title('Velocity Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    # Plot 3: Wall Shear Stress
    ax3 = axes[2]
    ax3.plot(timesteps, data['wss_mean'], 'g-', linewidth=2, label='Mean')
    ax3.plot(timesteps, data['wss_max'], 'g--', linewidth=1.5, alpha=0.7, label='Max')
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('WSS [kg/(mm·s²)]', fontsize=12)
    ax3.set_title('Wall Shear Stress Evolution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')

    plt.tight_layout()

    # Save figure
    output_file = f"{output_dir}/fluid_time_series_detailed.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")

    # Create second figure: Steady-state analysis (last 100 timesteps)
    if n_steps > 100:
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), dpi=150)

        start_idx = n_steps - 100
        ts_subset = timesteps[start_idx:]

        # Pressure
        ax1 = axes2[0]
        ax1.plot(ts_subset, data['pressure_mean'][start_idx:], 'b-', linewidth=2)
        ax1.fill_between(ts_subset,
                          data['pressure_min'][start_idx:],
                          data['pressure_max'][start_idx:],
                          alpha=0.3, color='blue')
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Pressure [kg/(mm·s²)]', fontsize=12)
        ax1.set_title('Pressure (Last 100 Timesteps)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Velocity
        ax2 = axes2[1]
        ax2.plot(ts_subset, data['velocity_mean'][start_idx:], 'r-', linewidth=2, label='Mean')
        ax2.plot(ts_subset, data['velocity_max'][start_idx:], 'r--', linewidth=1.5, alpha=0.7, label='Max')
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Velocity [mm/s]', fontsize=12)
        ax2.set_title('Velocity (Last 100 Timesteps)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        # WSS
        ax3 = axes2[2]
        ax3.plot(ts_subset, data['wss_mean'][start_idx:], 'g-', linewidth=2, label='Mean')
        ax3.plot(ts_subset, data['wss_max'][start_idx:], 'g--', linewidth=1.5, alpha=0.7, label='Max')
        ax3.set_xlabel('Timestep', fontsize=12)
        ax3.set_ylabel('WSS [kg/(mm·s²)]', fontsize=12)
        ax3.set_title('WSS (Last 100 Timesteps)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')

        plt.tight_layout()

        output_file2 = f"{output_dir}/fluid_steady_state.png"
        fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"Saved steady-state plot to {output_file2}")

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    # Overall statistics
    print("\nOverall (all timesteps):")
    print(f"  Pressure:  {np.mean(data['pressure_mean']):.3f} ± {np.std(data['pressure_mean']):.3f} kg/(mm·s²)")
    print(f"  Velocity:  {np.mean(data['velocity_mean']):.3f} ± {np.std(data['velocity_mean']):.3f} mm/s")
    print(f"  WSS:       {np.mean(data['wss_mean']):.6f} ± {np.std(data['wss_mean']):.6f} kg/(mm·s²)  [{np.mean(data['wss_mean'])*10000:.2f} dyne/cm²]")

    # Steady-state statistics (last 100 timesteps)
    if n_steps > 100:
        print("\nSteady-state (last 100 timesteps):")
        print(f"  Pressure:  {np.mean(data['pressure_mean'][-100:]):.3f} ± {np.std(data['pressure_mean'][-100:]):.3f} kg/(mm·s²)")
        print(f"  Velocity:  {np.mean(data['velocity_mean'][-100:]):.3f} ± {np.std(data['velocity_mean'][-100:]):.3f} mm/s")
        print(f"  WSS:       {np.mean(data['wss_mean'][-100:]):.6f} ± {np.std(data['wss_mean'][-100:]):.6f} kg/(mm·s²)  [{np.mean(data['wss_mean'][-100:])*10000:.2f} dyne/cm²]")

        # Check convergence (coefficient of variation in last 100 steps)
        cv_pressure = np.std(data['pressure_mean'][-100:]) / np.mean(data['pressure_mean'][-100:]) * 100
        cv_velocity = np.std(data['velocity_mean'][-100:]) / np.mean(data['velocity_mean'][-100:]) * 100
        cv_wss = np.std(data['wss_mean'][-100:]) / np.mean(data['wss_mean'][-100:]) * 100

        print("\nConvergence (Coefficient of Variation in last 100 steps):")
        print(f"  Pressure:  {cv_pressure:.2f}%")
        print(f"  Velocity:  {cv_velocity:.2f}%")
        print(f"  WSS:       {cv_wss:.2f}%")

        if cv_pressure < 1.0 and cv_velocity < 1.0:
            print("\n✓ Simulation appears to have reached steady state")
        else:
            print("\n⚠ Simulation may not be fully converged")

    print("="*60 + "\n")

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "post_results/fluid_results.csv"

    plot_fluid_results(csv_file)

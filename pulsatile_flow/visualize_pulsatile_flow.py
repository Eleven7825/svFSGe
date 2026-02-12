import numpy as np
import matplotlib.pyplot as plt

# Read the data file
data_file = 'pulsatile_flow.dat'

# Read data starting from the second row (skip first row with metadata)
data = np.loadtxt(data_file, skiprows=1)

# Extract time and velocity columns
time = data[:, 0]  # First column: time (s)
velocity = data[:, 1]  # Second column: velocity (mm/s)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the data
ax.plot(time, velocity, 'b-', linewidth=2, label='Velocity')
ax.grid(True, alpha=0.3)
ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Velocity (mm/s)', fontsize=12, fontweight='bold')
ax.set_title('Pulsatile Flow Velocity Profile', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

# Add some statistics to the plot
mean_vel = np.mean(velocity)
max_vel = np.max(velocity)
min_vel = np.min(velocity)

# Calculate time-averaged velocity (weighted by time intervals)
time_avg_vel = np.trapz(velocity, time) / (time[-1] - time[0])

stats_text = f'Time Average: {time_avg_vel:.2f} mm/s\nMax: {max_vel:.2f} mm/s\nMin: {min_vel:.2f} mm/s'
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure as PDF in post_results folder
import os
os.makedirs('post_results', exist_ok=True)
output_file = 'post_results/pulsatile_flow_visualization.pdf'
plt.savefig(output_file, format='pdf', bbox_inches='tight')
print(f"Figure saved as {output_file}")

# Display some basic statistics
print(f"\nData Statistics:")
print(f"  Number of time points: {len(time)}")
print(f"  Time range: {time[0]:.4f} to {time[-1]:.4f} s")
print(f"  Time duration: {time[-1] - time[0]:.4f} s")
print(f"  Time-averaged velocity: {time_avg_vel:.2f} mm/s")
print(f"  Max velocity: {max_vel:.2f} mm/s")
print(f"  Min velocity: {min_vel:.2f} mm/s")
print(f"  Velocity range: {max_vel - min_vel:.2f} mm/s")

# Show the plot
plt.show()

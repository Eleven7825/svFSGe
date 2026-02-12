import numpy as np

# Read the data file
data_file = 'pulsatile_flow.dat'

# Read the first line (metadata)
with open(data_file, 'r') as f:
    first_line = f.readline().strip()

# Read data starting from the second row (skip first row with metadata)
data = np.loadtxt(data_file, skiprows=1)

# Extract time and velocity columns
time = data[:, 0]  # First column: time (s)
velocity = data[:, 1]  # Second column: velocity (mm/s)

# Calculate time-averaged velocity (weighted by time intervals)
time_avg_vel = np.trapz(velocity, time) / (time[-1] - time[0])

print(f"Original time-averaged velocity: {time_avg_vel:.2f} mm/s")

# Calculate scaling factor to get average of -1000
target_avg = -1000.0
scaling_factor = target_avg / time_avg_vel

print(f"Scaling factor: {scaling_factor:.6f}")

# Scale the velocity values
velocity_normalized = velocity * scaling_factor

# Verify the new average
new_time_avg_vel = np.trapz(velocity_normalized, time) / (time[-1] - time[0])
print(f"New time-averaged velocity: {new_time_avg_vel:.2f} mm/s")

# Write the normalized data back to the file
with open(data_file, 'w') as f:
    # Write the metadata line
    f.write(first_line + '\n')
    # Write the normalized data
    for t, v in zip(time, velocity_normalized):
        f.write(f'{t:.10e} {v:.10e}\n')

print(f"\nNormalized data written to {data_file}")

#!/usr/bin/env python
"""
Run standalone fluid simulation (no FSG coupling)
"""
from svfsi import svFSI
from os.path import join
import subprocess
import sys

# Initialize and run setup (generates mesh, stages BC files)
print("Setting up simulation...")
sim = svFSI("in_sim/partitioned_full.json")

# Copy fluid mesh and boundary faces to expected locations
# (normally done by set_fluid during coupling)
import shutil
mesh_src = join(sim.p["f_out"], "mesh_tube_fsi/fluid/mesh-complete.mesh.vtu")
mesh_dst = join(sim.p["f_out"], "fluid.vtu")
shutil.copy(mesh_src, mesh_dst)
print(f"Copied mesh: {mesh_src} -> fluid.vtu")

# Copy boundary face files
for face in ["start", "end", "interface"]:
    src = join(sim.p["f_out"], f"mesh_tube_fsi/fluid/mesh-surfaces/{face}.vtp")
    dst = join(sim.p["f_out"], f"{face}.vtp")
    shutil.copy(src, dst)
    print(f"Copied face: {face}.vtp")

# Write constant outlet pressure BC file (normally done by set_fluid)
p = sim.p["fluid"]["p0"]
with open(join(sim.p["f_out"], "steady_pressure.dat"), "w") as f:
    f.write("2 1\n")
    f.write(f"0.0 {p}\n")
    f.write(f"9999999.0 {p}\n")
print(f"Created steady_pressure.dat (p={p})")

# Run fluid solver
print(f"\nRunning fluid simulation in {sim.p['f_out']}...")
print(f"Output will be saved to {sim.p['f_out']}/steady/")

exe = ["mpiexec", "-np", str(sim.p["n_procs"]["fluid"])]
exe += [sim.p["exe"]["fluid"]]
exe += [join("in_svfsi", "steady_full.xml")]

print(f"\nCommand: {' '.join(exe)}")
result = subprocess.run(exe, cwd=sim.p["f_out"])

if result.returncode == 0:
    print(f"\n✓ Simulation completed successfully")
    print(f"Results: {sim.p['f_out']}/steady/")
else:
    print(f"\n✗ Simulation failed with exit code {result.returncode}")
    sys.exit(result.returncode)

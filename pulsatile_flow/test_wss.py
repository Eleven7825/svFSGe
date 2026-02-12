#!/usr/bin/env python
import sys
import os
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk_functions import read_geo

# Read one VTU file
geo = read_geo('steady/steady_097.vtu').GetOutput()

# Check WSS array
print("=== Checking WSS array ===")
if geo.GetPointData().HasArray('WSS'):
    wss = v2n(geo.GetPointData().GetArray('WSS'))
    wss_mag = np.linalg.norm(wss, axis=1)
    print(f"WSS array exists: shape = {wss.shape}")
    print(f"WSS magnitude range: {np.min(wss_mag):.6e} to {np.max(wss_mag):.6e}")
    print(f"WSS mean magnitude: {np.mean(wss_mag):.6e}")
    print(f"Number of points with non-zero WSS: {np.sum(wss_mag > 1e-10)}/{len(wss_mag)}")
else:
    print("ERROR: WSS array not found!")

# Check point coordinates
pts = v2n(geo.GetPoints().GetData())
print(f"\nTotal points in mesh: {len(pts)}")

# Find points near the wall (inner radius)
r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
ri = np.min(r)
ro = np.max(r)
print(f"Radial range: {ri:.4f} to {ro:.4f} mm")

# Find wall points
wall_tolerance = 0.01
wall_pts = np.where(np.abs(r - ri) < wall_tolerance)[0]
print(f"Points near wall (r ~= {ri:.4f}): {len(wall_pts)}")

if len(wall_pts) > 0 and geo.GetPointData().HasArray('WSS'):
    wall_wss = wss_mag[wall_pts]
    print(f"WSS at wall: mean = {np.mean(wall_wss):.6e}, max = {np.max(wall_wss):.6e}")

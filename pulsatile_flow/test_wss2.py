#!/usr/bin/env python
import sys
import os
import numpy as np
sys.path.insert(0, '.')

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk_functions import read_geo

# Read one VTU file
geo = read_geo('steady/steady_097.vtu').GetOutput()
pts = v2n(geo.GetPoints().GetData())

# Get WSS
wss = v2n(geo.GetPointData().GetArray('WSS'))
wss_mag = np.linalg.norm(wss, axis=1)

# Find where WSS is non-zero
nonzero_wss = wss_mag > 1e-10
wss_pts = pts[nonzero_wss]

print(f"Total points: {len(pts)}")
print(f"Points with non-zero WSS: {np.sum(nonzero_wss)}")

# Check radial distribution of WSS points
r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
r_wss = r[nonzero_wss]

print(f"\nAll points: r from {np.min(r):.4f} to {np.max(r):.4f} mm")
print(f"WSS points: r from {np.min(r_wss):.4f} to {np.max(r_wss):.4f} mm")

# Check unique radii where WSS exists
unique_r_wss = np.unique(np.round(r_wss, 4))
print(f"\nUnique radii with WSS: {unique_r_wss[:10]}")

# WSS should be at the OUTER wall boundary, not inner!
print(f"\nWSS appears to be at radius: {np.max(r_wss):.4f} mm")

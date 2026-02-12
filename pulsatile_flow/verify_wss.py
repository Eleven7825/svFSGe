#!/usr/bin/env python
"""Quick verification that WSS is extracted correctly"""
import sys
import os
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk_functions import read_geo
from post_steady_fluid import xyz2cra, get_ids, extract_wss

# Read one VTU file
geo = read_geo('steady/steady_097.vtu').GetOutput()
pts = v2n(geo.GetPoints().GetData())

# Get interface IDs (at outer wall where WSS is defined)
ids_interface, coords_interface = get_ids(pts, 'interface')

print("=== WSS Extraction Verification ===\n")
print(f"Interface locations found: {len(ids_interface)}")

# Extract WSS
post = {loc: defaultdict(list) for loc in ids_interface.keys()}
extract_wss(post, geo, pts, ids_interface)

# Check a few locations
test_locs = [(0, 'wall', ':'), (':', 'wall', 'mid'), (3, 'wall', ':'), (6, 'wall', ':')]

for loc in test_locs:
    if loc in post and 'wss' in post[loc] and len(post[loc]['wss']) > 0:
        wss_data = np.array(post[loc]['wss'][0])
        print(f"{loc}:")
        print(f"  Points: {len(wss_data)}")
        print(f"  WSS range: {np.min(wss_data):.6f} to {np.max(wss_data):.6f} kg/(mm·s²)")
        print(f"  WSS mean: {np.mean(wss_data):.6f} kg/(mm·s²)")
        print(f"  WSS mean (dyne/cm²): {np.mean(wss_data)*10000:.2f}\n")
    else:
        print(f"{loc}: No data\n")

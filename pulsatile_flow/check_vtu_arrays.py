#!/usr/bin/env python
"""
Quick script to check what arrays are available in VTU files
"""

import glob
from vtk_functions import read_geo
from vtk.util.numpy_support import vtk_to_numpy as v2n

# Find first VTU file
vtu_files = sorted(glob.glob("steady/steady_*.vtu"))
if not vtu_files:
    print("No VTU files found in steady/")
    exit(1)

print(f"Checking file: {vtu_files[0]}")
print("=" * 60)

# Read the file
geo = read_geo(vtu_files[0]).GetOutput()

# Check point data arrays
print("\nPoint Data Arrays:")
print("-" * 60)
point_data = geo.GetPointData()
for i in range(point_data.GetNumberOfArrays()):
    array_name = point_data.GetArrayName(i)
    array = point_data.GetArray(i)
    n_components = array.GetNumberOfComponents()
    n_tuples = array.GetNumberOfTuples()
    print(f"  {array_name}: {n_components} components, {n_tuples} points")

# Check cell data arrays
print("\nCell Data Arrays:")
print("-" * 60)
cell_data = geo.GetCellData()
for i in range(cell_data.GetNumberOfArrays()):
    array_name = cell_data.GetArrayName(i)
    array = cell_data.GetArray(i)
    n_components = array.GetNumberOfComponents()
    n_tuples = array.GetNumberOfTuples()
    print(f"  {array_name}: {n_components} components, {n_tuples} cells")

# Number of points and cells
print("\nMesh Info:")
print("-" * 60)
print(f"  Number of points: {geo.GetNumberOfPoints()}")
print(f"  Number of cells: {geo.GetNumberOfCells()}")

print("\n" + "=" * 60)
print(f"Total VTU files found: {len(vtu_files)}")

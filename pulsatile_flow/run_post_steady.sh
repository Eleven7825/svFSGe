#!/bin/bash
# Wrapper script to run post_steady_fluid.py with pvpython

PVPYTHON="/home/shiyi/ParaView-5.13.0-RC1-MPI-Linux-Python3.10-x86_64/bin/pvpython"

# Run the post-processing script
$PVPYTHON post_steady_fluid.py "$@" 2>&1 | grep -v "hwloc/linux"

#!/bin/bash
#SBATCH --job-name=iqn_osi_add
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# iqn_ils coupling (the standard, non-compressed algorithm) with the G&R
# stimulus replaced by OSI, using the additive+tanh stimulus form
# (gr_full_restart_additive_osi.xml). Same mesh/fluid/solid config as
# partitioned_pulsatile_uber_robin_osi_additive.json -- only coup.method
# differs (uber_robin -> iqn_ils) -- for a same-stimulus, coupling-
# algorithm-only comparison.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_iqn_ils_osi_additive.json

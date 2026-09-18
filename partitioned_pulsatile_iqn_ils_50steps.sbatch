#!/bin/bash
#SBATCH --job-name=iqn_50s_tol3
#SBATCH --partition=week
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# Coarse mesh, PULSATILE flow (2 cycles x 50 steps/cycle, down from 200),
# default iqn_ils coupling, coup.tol=1e-3 (default). Companion job
# partitioned_pulsatile_iqn_ils_50steps_tol1e-5.sbatch is identical except
# coup.tol=1e-5, to compare how a tighter coupling tolerance affects
# convergence/results at the new reduced pulsatile step count.
#
# No srun: srun/interactive jobs are QOS-capped at 4 concurrent per user
# and not even allowed on the week partition, but sbatch jobs get "normal"
# QOS regardless. This job just runs its own command directly.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_iqn_ils_50steps.json

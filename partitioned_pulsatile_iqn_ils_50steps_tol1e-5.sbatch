#!/bin/bash
#SBATCH --job-name=iqn_50s_tol5
#SBATCH --partition=week
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# Same as partitioned_pulsatile_iqn_ils_50steps.sbatch (coarse mesh,
# pulsatile 2 cycles x 50 steps/cycle, iqn_ils coupling) but with
# coup.tol tightened from the default 1e-3 to 1e-5.
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
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_iqn_ils_50steps_tol1e-5.json

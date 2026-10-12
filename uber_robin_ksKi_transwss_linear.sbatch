#!/bin/bash
#SBATCH --job-name=ur_ksKi_lin
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# Same as uber_robin_ksKi_transwss.sbatch (loss-of-mechanosensing driven by
# the local TransWSS/WSS fraction, main stimulus reverted to plain WSS), but
# using the plain linear form KsKi *= 1 - trans_sense (ksKi_transwss_linear,
# ksKi_transwss_scale=1.0) instead of the tanh-saturating one.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_uber_robin_ksKi_transwss_linear.json

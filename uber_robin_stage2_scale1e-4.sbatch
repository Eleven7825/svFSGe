#!/bin/bash
#SBATCH --job-name=ur_s2_1e-4
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# Stage 2 of the elastin-then-mechanosensing experiment: continues from
# partitioned_pulsatile_uber_robin_elastin_only_mag05's final converged
# state (uber_robin_checkpoint) -- elastin held fixed at that stable,
# fully-converged degraded state (n_t_pre=0, n_t_end=1) -- with loss of
# mechanosensing now active from the start (ksKi_transwss_gate_step=0),
# tanh form, scale=1e-4. Part of a scale sweep (1e-4, 1e-3, 1e-2)
# bracketing the actual observed stable-regime trans_sense range.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_uber_robin_stage2_scale1e-4.json

#!/bin/bash
#SBATCH --job-name=ur_elastin1
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# Stage 1 of a two-run elastin-then-mechanosensing experiment: elastin
# degradation ALONE (ksKi_transwss_gate_step set far beyond this run's own
# step range, so KsKi is left completely untouched throughout -- full
# mechanosensing intact), the full standard uber_robin schedule (n_steps=10,
# final step gets the proper hybrid Aitken->IQN-ILS treatment). Its final
# converged state becomes stage 2's (uber_robin_elastin_then_ksKi.sbatch)
# starting point via run()'s uber_robin_checkpoint mechanism.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_uber_robin_elastin_only.json

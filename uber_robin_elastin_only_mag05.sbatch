#!/bin/bash
#SBATCH --job-name=ur_elas_m05
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# Same as uber_robin_elastin_only.sbatch (elastin degradation alone, full
# mechanosensing intact) but with insult_mag reduced from the default 0.7 to
# 0.5 -- a gentler elastin insult, testing whether a genuinely stable
# elastin-only baseline exists at this lower severity (the default produced
# the same t=8 Negative Jacobian crash as every combined-mechanism run).

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_uber_robin_elastin_only_mag05.json

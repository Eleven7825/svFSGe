#!/bin/bash
#SBATCH --job-name=ur_ksKi_tws
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G

# uber_robin with the MAIN growth stimulus reverted to plain averaged WSS
# (tau/tauo ratio, the physiologically well-established mechanism), but
# loss-of-mechanosensing now driven by the LOCAL, instantaneous TransWSS/WSS
# fraction (ksKi_from_transwss in gr_full_restart_ksKi_transwss.xml) instead
# of the fixed geometric insult shape -- following Pfaller et al. 2024's own
# K_tausigma -> 0 mechanism (Section 4) but with a physiologically-motivated
# driver (Warboys et al. 2021; Ghim et al. 2017: endothelial mechanotransmission
# degrades under multidirectional/transverse shear) in place of a fixed mask.
# Elastin degradation is unaffected -- still the same geometric insult shape.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_pulsatile_uber_robin_ksKi_transwss.json

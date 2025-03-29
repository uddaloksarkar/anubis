#!/bin/bash

#SBATCH --job-name=j-plag
#SBATCH --output=j-dnf-%j.out
#SBATCH --error=j-dnf-%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=compute_full_node
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anubis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

module load cuda/12.2.2

module load anaconda3
source activate plag
cd ${SLURM_SUBMIT_DIR}
[ -d log ] || mkdir log

python3 execute-run.py  --offset 50

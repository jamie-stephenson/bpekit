#!/bin/bash
#SBATCH --job-name=maturin_develop
#SBATCH --partition=universe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ubuntu/shared/slurmlogs/%j_maturin_develop.log

srun bash -c "source ~/envs/bpekit/bin/activate && maturin develop"

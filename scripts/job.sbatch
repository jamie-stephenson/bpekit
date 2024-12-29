#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=universe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=1:00:00
#SBATCH --output=/home/ubuntu/shared/slurmlogs/%j_tokenize.log

source ~/envs/bpekit/bin/activate

# Train a tokenizer on a dataset AND use it to 
# encode that same dataset  
mpirun --bind-to none bpekit train data/wiki/raw/ 8132
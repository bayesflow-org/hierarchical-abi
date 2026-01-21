#!/bin/bash
#SBATCH --job-name stan_ar1
#SBATCH --output log/log_stan_ar1.%j.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 64
#SBATCH --partition intelsr_medium
#SBATCH --time 1-00:00:00
#SBATCH --array=0-0
#SBATCH --depend=afterany:20608877
#SBATCH --mail-type=END
#SBATCH --mail-user=jonas.arruda@uni-bonn.de

TQDM_DISABLE=1
export TQDM_DISABLE
export KERAS_BACKEND=torch

source /home/jarruda_hpc/hierarchical-abi/env_marvin.sh

python3.11 /home/jarruda_hpc/hierarchical-abi/experiments/run_stan.py
#python3.11 /home/jarruda_hpc/hierarchical-abi/experiments/gaussian_flat_score_matching.py 1
#python3.11 /home/jarruda_hpc/hierarchical-abi/experiments/gaussian_flat_score_matching.py 100
#python3.11 /home/jarruda_hpc/hierarchical-abi/experiments/ar\(1\)_score_matching.py 1
#python3.11 /home/jarruda_hpc/hierarchical-abi/experiments/ar\(1\)_score_matching.py 100
#python3.11 /home/jarruda_hpc/hierarchical-abi/experiments/fli\ score\ matching.py

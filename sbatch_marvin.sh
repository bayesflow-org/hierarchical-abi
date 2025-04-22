#!/bin/bash
#SBATCH --job-name hierarchical-ar1-stan
#SBATCH --output log/log_hierarchical_ar1_stan.%j.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 100GB
#SBATCH --partition mlgpu_medium
#SBATCH --gpus 1
#SBATCH --time 1-00:00:00
#SBATCH --array=40-49
#SBATCH --depend=afterany:2041138
#SBATCH --mail-type=END
#SBATCH --mail-user=jonas.arruda@uni-bonn.de

TQDM_DISABLE=1
export TQDM_DISABLE
export KERAS_BACKEND=torch

source /home/jarruda_hpc/hierarchical-abi/env_marvin.sh

#python3.11 /home/jarruda_hpc/hierarchical-abi/gaussian_flat_score_matching.py 1
#python3.11 /home/jarruda_hpc/hierarchical-abi/gaussian_flat_score_matching.py 100
python3.11 /home/jarruda_hpc/hierarchical-abi/ar\(1\)_score_matching.py 1
#python3.11 /home/jarruda_hpc/hierarchical-abi/ar\(1\)_score_matching.py 100
#python3.11 /home/jarruda_hpc/hierarchical-abi/fli\ score\ matching.py 

#!/bin/bash
#SBATCH --job-name hierarchical-abi-ar1
#SBATCH --output log/log_hierarchical_abi_ar1.%j.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --partition gn
#SBATCH --gpus 1
#SBATCH --time 3-00:00:00
#SBATCH --array=0-29%3
#SBATCH --depend=afterany:2063854
#SBATCH --mail-type=END
#SBATCH --mail-user=jonas.arruda@uni-bonn.de

TQDM_DISABLE=1
export TQDM_DISABLE

source /home/jonas/hierarchical-abi/env.sh

#python3.11 /home/jonas/hierarchical-abi/gaussian_flat_score_matching.py 1
#python3.11 /home/jonas/hierarchical-abi/gaussian_flat_score_matching.py 100
python3.11 /home/jonas/hierarchical-abi/ar\(1\)_score_matching.py 1
#python3.11 /home/jonas/hierarchical-abi/ar\(1\)_score_matching.py 100

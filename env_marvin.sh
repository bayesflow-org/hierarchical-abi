module purge

module load Miniforge3
module load CUDA/12.6.0

source ~/.bashrc
conda deactivate
conda activate habi


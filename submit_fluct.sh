#!/bin/bash
#SBATCH --job-name=fluct_sk
#SBATCH --partition=v100-al9_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=fluct_sk_%j.out
#SBATCH --error=fluct_sk_%j.err

# ============================================================
# Environment
# ============================================================

module purge
module load cuda/12.6.0

source ~/.bashrc
conda activate data_analysis

# ============================================================
# Run
# ============================================================

python fluctuation_Sk_factor.py

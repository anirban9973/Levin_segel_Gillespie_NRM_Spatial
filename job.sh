#!/bin/bash
#SBATCH --job-name=turing_model
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --partition=edr1-al9_large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-100

mkdir -p logs

start=$(date +%s)
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started at: $(date)"

# Each job runs 2400 realizations via OpenMP (24 threads x 100 sequential each)
# Seeds: job 1 -> 0..2399, job 2 -> 2400..4799, ..., job 20 -> 45600..47999
./a.out $SLURM_ARRAY_TASK_ID

end=$(date +%s)
elapsed=$(( end - start ))

days=$(( elapsed / 86400 ))
hours=$(( (elapsed % 86400) / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} finished at: $(date)"
echo "Elapsed time: ${days}d ${hours}h ${minutes}m ${seconds}s"

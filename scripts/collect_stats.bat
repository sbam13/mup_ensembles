#!/bin/bash

#SBATCH --job-name=collect_stats
#SBATCH -e /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.err
#SBATCH -o /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.out
#SBATCH --time=3-00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --partition=kempner,seas_gpu,gpu
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue

module load cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 Anaconda3/2020.11

source activate jupyter_39

cd /n/home07/ssainathan/workplace/gpu_scheduler/scripts

srun python3 collect_stats.py
#!/bin/bash

#SBATCH --job-name=lr1e-3
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --partition=pehlevan_gpu
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:1

module load cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 Anaconda3/2020.11

source activate gs

cp -R /n/home07/ssainathan/workplace/gpu_scheduler /tmp
cd /tmp/gpu_scheduler

rm conf/config.yaml

printf "defaults:\n  - experiment: sweep_dataset_size_16384_lr_0.001" > conf/config.yaml

chmod +x ./main.py

srun ./main.py
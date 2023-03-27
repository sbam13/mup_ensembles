#!/bin/bash

#SBATCH --job-name=256
#SBATCH -e /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.err
#SBATCH -o /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.out
#SBATCH --time=7-00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --partition=kempner
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:1
#SBATCH --requeue

module load cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 Anaconda3/2020.11

source activate jupyter_39

rm -rf /tmp/256
mkdir /tmp/256

mkdir /tmp/256/data-dir

mkdir /tmp/256/results

cp -R /n/home07/ssainathan/workplace/gpu_scheduler /tmp/256
cd /tmp/256/gpu_scheduler

rm conf/config.yaml

printf "defaults:\n  - experiment: sweep_256" > conf/config.yaml

srun python3 main.py
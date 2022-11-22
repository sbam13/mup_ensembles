#!/bin/bash

#SBATCH --job-name=8
#SBATCH -e /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.err
#SBATCH -o /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.out
#SBATCH --time=2-00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --partition=pehlevan_gpu,gpu
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:4
#SBATCH --requeue

module load cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 Anaconda3/2020.11

source activate gs

rm -rf /tmp/8
mkdir /tmp/8

mkdir /tmp/8/data-dir
cp -R "/n/holystore01/LABS/pehlevan_lab/Users/sab/cifar-10-batches-py" /tmp/8/data-dir

mkdir /tmp/8/results

cp -R /n/home07/ssainathan/workplace/gpu_scheduler /tmp/8
cd /tmp/8/gpu_scheduler

rm conf/config.yaml

printf "defaults:\n  - experiment: sweep_8" > conf/config.yaml

srun python3 main.py
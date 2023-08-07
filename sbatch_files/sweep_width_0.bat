#!/bin/bash

#SBATCH --job-name=0
#SBATCH -e /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.err
#SBATCH -o /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.out
#SBATCH --time=3-00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100


module load cuda/12.0.1-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01 python/3.10.9-fasrc01

source activate jax_env

rm -rf /tmp/0
mkdir /tmp/0

mkdir /tmp/0/data-dir

mkdir /tmp/0/results

cp -R /n/home07/ssainathan/workplace/mup_ensembles /tmp/0
cd /tmp/0/mup_ensembles

rm conf/config.yaml

printf "defaults:\n  - experiment: sweep_width_0" > conf/config.yaml

srun python3 main.py
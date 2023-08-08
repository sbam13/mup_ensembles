#!/bin/bash

#SBATCH --job-name=collect_stats
#SBATCH -e /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.err
#SBATCH -o /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.out
#SBATCH --time=3-00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue

module load cuda/12.0.1-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01 python/3.10.9-fasrc01

source activate jupyter_39

cd /n/home07/ssainathan/workplace/mup_ensembles/scripts

srun python3 ensemble_size_loss_data.py "/n/pehlevan_lab/Users/sab/ecd_final" "/n/pehlevan_lab/Users/sab/ensemble_size_loss_correspondence"
SBATCH_TEMPLATE = '''#!/bin/bash

#SBATCH --job-name={id}
#SBATCH -e /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.err
#SBATCH -o /n/home07/ssainathan/workplace/sbatch_out/slurm-%j.out
#SBATCH --time=3-00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --partition=kempner,seas_gpu,gpu
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue

module load cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 Anaconda3/2020.11

source activate jupyter_39

rm -rf /tmp/{id}
mkdir /tmp/{id}

mkdir /tmp/{id}/data-dir

mkdir /tmp/{id}/results

cp -R /n/home07/ssainathan/workplace/gpu_scheduler /tmp/{id}
cd /tmp/{id}/gpu_scheduler

rm conf/config.yaml

printf "defaults:\\n  - experiment: sweep_width_{id}" > conf/config.yaml

srun python3 main.py'''
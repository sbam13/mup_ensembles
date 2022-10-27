#!/bin/bash

#SBATCH --job-name=gpu
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --time=01:00:00  
#SBATCH --nodes=1        
#SBATCH --partition=pehlevan_gpu
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:4

@REM loads CUDA and conda
module load cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 Anaconda3/2020.11

srun source activate 

@REM rem copies cifar data to scratch for fast access
@REM CIFAR_FOLDER="/n/holystore01/LABS/pehlevan_lab/Users/sab/cifar-10-batches-py"
@REM LOCAL_CIFAR_FOLDER = "/tmp/cifar10"


@REM mkdir $LOCAL_CIFAR_FOLDER
@REM cp -R $CIFAR_LOC $LOCAL_CIFAR_FOLDER

@REM LOCAL_SAVE_FOLDER="/tmp/results"
@REM mkdir $LOCAL_SAVE_FOLDER

@REM rem TODO move
@REM cp -R $LOCAL_SAVE_FOLDER 

# Add lines here to run your GPU-based computations.

#!/bin/bash

rem TODO amend
#SBATCH --job-name=gpu
#SBATCH --output=gpu.out 
#SBATCH --error=gpu.err  
#SBATCH --time=01:00:00  
#SBATCH --nodes=1        
#SBATCH --partition=pehlevan_gpu
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:4

rem loads CUDA and conda
module load cuda/11.7.1-fasrc01 cudnn/8.5.0.96_cuda11-fasrc01 Anaconda3/2020.11

rem copies cifar data to scratch for fast access
CIFAR_FOLDER="/n/holystore01/LABS/pehlevan_lab/Users/sab/cifar-10-batches-py"
LOCAL_CIFAR_FOLDER = "/tmp/cifar10"


mkdir $LOCAL_CIFAR_FOLDER
cp -R $CIFAR_LOC $LOCAL_CIFAR_FOLDER

LOCAL_SAVE_FOLDER="/tmp/results"
mkdir $LOCAL_SAVE_FOLDER

rem TODO move
cp -R $LOCAL_SAVE_FOLDER 

# Add lines here to run your GPU-based computations.

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

# Add lines here to run your GPU-based computations.

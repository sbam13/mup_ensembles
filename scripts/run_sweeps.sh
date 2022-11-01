for sweep in {0..$1}
do
    sbatch "/n/home07/ssainathan/workplace/gpu_scheduler/sbatch_files/sweep_${sweep}.yaml"
done
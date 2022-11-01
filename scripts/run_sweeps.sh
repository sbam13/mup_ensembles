for sweep in {0..8}
do
    sbatch "/n/home07/ssainathan/workplace/gpu_scheduler/sbatch_files/sweep_${sweep}.bat"
done
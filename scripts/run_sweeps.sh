for sweep in {0..32}
do
    sbatch "/n/home07/ssainathan/workplace/gpu_scheduler/sbatch_files/sweep_width_${sweep}.bat"
done
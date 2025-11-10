for denoiser in ["linear","linear_text"]:
    command=f"sbatch -J weights --err=slurm_chip/trying/{denoiser}.err --out=slurm_chip/trying/{denoiser}.out "
    command+=f" runpygpu_chip.sh diffusion.py --name jlbaker361/weights-{denoiser}-try --denoiser {denoiser} --val_interval 10 --epochs 10 "
    command+=f" --limit 100 --gradient_accumulation_steps 8  "
    print(command)
for denoiser in ["linear","linear_text"]:
    command=f"sbatch -J weights --err=slurm_chip/training/{denoiser}.err --out=slurm_chip/training/{denoiser}.out "
    command+=f" runpygpu_chip.sh diffusion.py --name jlbaker361/weights-{denoiser} --denoiser {denoiser} "
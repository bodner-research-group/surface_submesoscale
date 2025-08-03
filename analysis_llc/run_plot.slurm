#!/bin/bash
#SBATCH --job-name=avg_12h
#SBATCH --output=avg_12h.out
#SBATCH --error=avg_12h.err
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1                # IMPORTANT: use 1 task (you use 64 threads inside Dask)
#SBATCH --cpus-per-task=64        # Explicitly request 64 CPUs for Dask
#SBATCH --time=10:00:00
#SBATCH --mem=386G

source ~/.bashrc-conda

python3 ts1_calc_12h_avg.py


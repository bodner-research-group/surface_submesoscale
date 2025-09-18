#!/bin/bash
#SBATCH --job-name=llc2swot_cycles_15
#SBATCH --output=llc2swot_cycles_15.out
#SBATCH --error=llc2swot_cycles_15.err
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1                # IMPORTANT: use 1 task (you use 64 threads inside Dask)
#SBATCH --cpus-per-task=20        # Explicitly request 64 CPUs for Dask
#SBATCH --time=3:00:00
#SBATCH --mem=386G

source ~/.bashrc-conda

python3 llc2swot_sbatch.py
echo "Complete: llc2swot_cycles_15"

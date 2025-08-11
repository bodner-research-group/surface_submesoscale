#!/bin/bash
#SBATCH --job-name=tropics
#SBATCH --output=tropics.out
#SBATCH --error=tropics.err
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1                # IMPORTANT: use 1 task (you use 64 threads inside Dask)
#SBATCH --cpus-per-task=64        # Explicitly request 64 CPUs for Dask
#SBATCH --time=12:00:00
#SBATCH --mem=386G

source ~/.bashrc-conda

python3 ts1_24h_avg_TSW.py
echo "Complete: ts1_24h_avg_TSW"

python3 ts1_24h_avg_surfaceUV.py
echo "Complete: ts1_24h_avg_surfaceUV"

python3 ts2-1_7day_avg.py
echo "Complete: ts2-1_7day_avg"

python3 ts2-2_mean_Hml.py
echo "Complete: ts2-2_mean_Hml"

python3 ts3-1_wb_cros.py
echo "Complete: ts3-1_wb_cros"

python3 ts3-2_plot_weekly_wb.py
echo "Complete: ts3-2_plot_weekly_wb"

python3 ts5_Qnet_mean.py
echo "Complete: ts5_Qnet_mean"


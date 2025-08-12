#!/bin/bash
#SBATCH --job-name=tropics
#SBATCH --output=tropics_plot.out
#SBATCH --error=tropics_plot.err
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1                # IMPORTANT: use 1 task (you use 64 threads inside Dask)
#SBATCH --cpus-per-task=64        # Explicitly request 64 CPUs for Dask
#SBATCH --time=12:00:00
#SBATCH --mem=386G

source ~/.bashrc-conda

# python3 ts1_24h_avg_TSW.py
# echo "Complete: ts1_24h_avg_TSW"

# python3 ts2_24h_avg_surfaceUV.py
# echo "Complete: ts2_24h_avg_surfaceUV"

# python3 ts3_7day_avg.py 
# echo "Complete: ts3_7day_avg"

# python3 ts7_Qnet_mean_dask.py
# echo "Complete: ts7_Qnet_mean_dask"

python3 ts4_Hml_mean.py
echo "Complete: ts4_Hml_mean"

python3 ts5_wb_cros_calc.py
echo "Complete: ts5_wb_cros_calc"

python3 ts6_wb_cros_plot.py
echo "Complete: ts6_wb_cros_plot"

# python3 ts8_vorticity_strain_calc.py
# echo "Complete: ts8_vorticity_strain_calc"

# python3 ts9_vorticity_strain_plot.py
# echo "Complete: ts9_vorticity_strain_plot"

# python3 ts10_calc_Tu_vertical.py
# echo "Complete: ts10_calc_Tu_vertical"

# python3 ts11_calc_Tu_horizontal.py
# echo "Complete: ts11_calc_Tu_horizontal"
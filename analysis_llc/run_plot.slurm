#!/bin/bash
#SBATCH --job-name=hourly_wb_calc42
#SBATCH --output=hourly_wb_calc42.out
#SBATCH --error=hourly_wb_calc42.err
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1                # IMPORTANT: use 1 task (you use 64 threads inside Dask)
#SBATCH --cpus-per-task=64        # Explicitly request 64 CPUs for Dask
#SBATCH --time=12:00:00
#SBATCH --mem=386G

source ~/.bashrc-conda

python3 py31_hourly_wb_calc42.py
echo "Complete: py31_hourly_wb_calc42"

# python3 py31_hourly_rho5.py
# echo "Complete: py31_hourly_rho5"

# python3 py1_24h_avg_TSW.py
# echo "Complete: py1_24h_avg_TSW"

# python3 py2_24h_avg_surface.py
# echo "Complete: py2_24h_avg_surface"

# python3 py3_7day_rolling_mean.py 
# echo "Complete: py3_7day_rolling_mean"

# python3 py7_Qnet_mean.py
# echo "Complete: py7_Qnet_mean"

# python3 py4_Hml_mean.py
# echo "Complete: py4_Hml_mean"

# python3 py5_wb_cros_calc_7d_rolling.py
# echo "Complete: py5_wb_cros_calc_7d_rolling"

# python3 py6_wb_cros_plot.py
# echo "Complete: py6_wb_cros_plot"


# python3 py8_vorticity_strain_calc.py
# echo "Complete: py8_vorticity_strain_calc"

# python3 py9_vorticity_strain_plot.py
# echo "Complete: py9_vorticity_strain_plot"

# python3 py10_TurnerAngle_calc.py
# echo "Complete: py10_TurnerAngle_calc"

# python3 py11_TurnerAngle_plot.py
# echo "Complete: py11_TurnerAngle_plot"


# python3 py12_SSH_gradient.py
# echo "Complete: py12_SSH_gradient"
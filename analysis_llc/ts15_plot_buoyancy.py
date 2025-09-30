
import xarray as xr
import numpy as np
import os
from xgcm import Grid
import gsw

from set_constant import domain_name, face, i, j

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from set_colormaps import WhiteBlueGreenYellowRed
cmap = WhiteBlueGreenYellowRed()

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# ========= Paths =========
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
input_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/"
input_file = os.path.join(input_path, "surfaceT-S-rho-buoy-alpha-beta_gradient_magnitude.nc")

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}"

ds = xr.open_dataset(input_file)

    # "buoy_s_grad_mag_daily": buoy_s_grad_mag_daily,
    # "buoy_s_grad_mag_weekly": buoy_s_grad_mag_weekly,
    # "buoy_s_linear_grad_mag_daily": buoy_s_linear_grad_mag_daily,
    # "buoy_s_linear_grad_mag_weekly": buoy_s_linear_grad_mag_weekly,

buoy_s_grad_mag_daily = ds.buoy_s_grad_mag_daily
buoy_s_linear_grad_mag_daily = ds.buoy_s_linear_grad_mag_daily

tt_s_grad_mag_daily = ds.tt_s_grad_mag_daily
ss_s_grad_mag_daily = ds.ss_s_grad_mag_daily
rho_s_grad_mag_daily = ds.rho_s_grad_mag_daily
alpha_s_daily = ds.alpha_s_daily
beta_s_daily = ds.beta_s_daily

# Compute required quantities
gravity = 9.81
rho0 = 999.8
# buoy_s_grad_mag_daily = gravity*rho_s_grad_mag_daily/rho0

g_alpha_gradT = gravity*alpha_s_daily * tt_s_grad_mag_daily
g_beta_gradS = gravity*beta_s_daily * ss_s_grad_mag_daily

b_linear_grad = buoy_s_linear_grad_mag_daily
# b_linear_grad = g_alpha_gradT - g_beta_gradS

# Time values
time = ds.time

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(time, buoy_s_grad_mag_daily, label='|∇b| (nonlinear EOS)', color='black', linewidth=2)
ax.plot(time, b_linear_grad, label='|∇b| (linear EOS)', color='red', linestyle='--')
ax.plot(time, g_alpha_gradT, label='gα|∇T|', color='blue', linestyle=':')
ax.plot(time, g_beta_gradS, label='gβ|∇S|', color='green', linestyle='-.')

# Axis formatting
ax.set_ylabel("Buoyancy Gradient Magnitude [s⁻²]")
ax.set_xlabel("Time")
ax.set_title("Surface Buoyancy Gradient Time Series")
ax.legend(loc='upper left')

# Improve time axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks every month
fig.autofmt_xdate()

# Grid lines
ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.8)
ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)  # Minor grid lines

# Save figure
os.makedirs(figdir, exist_ok=True)
fig.savefig(os.path.join(figdir, "surface_buoyancy_gradient_time_series.png"), dpi=300, bbox_inches='tight')

plt.show()



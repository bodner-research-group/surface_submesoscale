
import xarray as xr
import numpy as np
import os
from xgcm import Grid
import gsw

# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

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
rho0 = 1027.5
# buoy_s_grad_mag_daily = gravity*rho_s_grad_mag_daily/rho0

tAlpha_ref = 2.0e-4
sBeta_ref = 7.4e-4

# g_alpha_gradT = gravity*alpha_s_daily * tt_s_grad_mag_daily
# g_beta_gradS = gravity*beta_s_daily * ss_s_grad_mag_daily
g_alpha_gradT = gravity*tAlpha_ref * tt_s_grad_mag_daily
g_beta_gradS = gravity*sBeta_ref * ss_s_grad_mag_daily

b_linear_grad = buoy_s_linear_grad_mag_daily
# b_linear_grad = g_alpha_gradT - g_beta_gradS

# Time values
time = ds.time

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(time, buoy_s_grad_mag_daily, label='|∇b| (nonlinear EOS)', color='black', linewidth=2)
ax.plot(time, b_linear_grad, label='|∇b| (linear EOS)', color='red', linestyle='--')
ax.plot(time, g_alpha_gradT, label='gα₀|∇T|', color='blue', linestyle=':')
ax.plot(time, g_beta_gradS, label='gβ₀|∇S|', color='green', linestyle='-.')

# Axis formatting
ax.set_ylabel("Buoyancy Gradient Magnitude [s⁻²]")
ax.set_xlabel("Time")
ax.set_title("Surface Buoyancy Gradient Time Series")
ax.legend(loc='upper left')

# Improve time axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks every month
# fig.autofmt_xdate()

# Grid lines
ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.8)
ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)  # Minor grid lines

# Save figure
os.makedirs(figdir, exist_ok=True)
fig.savefig(os.path.join(figdir, "surface_buoyancy_gradient_time_series.png"), dpi=300, bbox_inches='tight')

plt.show()



# Create two vertically stacked subplots
fig2, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True)

# Plot α (thermal expansion)
ax1.plot(time, alpha_s_daily, label=r'$\alpha$ (thermal expansion)', color='orange')
ax1.set_ylabel(r"$\alpha$ [$^\circ$C$^{-1}$]")
ax1.set_title("Daily Time Series of Surface α (thermal expansion)")
ax1.grid(True)
ax1.legend()
ax1.set_ylim(1.3e-4, 2.1e-4)  # Example limits — adjust as needed

# Plot β (haline contraction)
ax2.plot(time, beta_s_daily, label=r'$\beta$ (haline contraction)', color='purple')
ax2.set_ylabel(r"$\beta$ [psu$^{-1}$]")
ax2.set_xlabel("Time")
ax2.set_title("Daily Time Series of Surface β (haline contraction)")
ax2.grid(True)
ax2.legend()
ax2.set_ylim(6.5e-4, 8.5e-4)  # Example limits — adjust as needed

# Format x-axis (shared)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_minor_locator(mdates.MonthLocator())

# Save figure
fig2.tight_layout()
fig2.savefig(os.path.join(figdir, "alpha_beta_time_series_subplots.png"), dpi=300, bbox_inches='tight')
plt.show()

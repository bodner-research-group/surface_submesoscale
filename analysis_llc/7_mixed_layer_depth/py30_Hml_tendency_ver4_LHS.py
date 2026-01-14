####### Add surface buoyancy to the left hand side


import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

# ========== Domain ==========
from set_constant import domain_name, face, i, j

##### Load LLC dataset
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lat = ds1['YC'].isel(face=face, i=i, j=j)
Coriolis = 4*np.pi/86164*np.sin(lat*np.pi/180)
abs_f = np.abs(Coriolis.mean(dim=("i","j")).values)

##### Constants
g = 9.81
rho0 = 1027.5
delta_rho = 0.03
Ce = 0.06
const = 1

# ==============================================================
# 1. dHml/dt from MLD
# ==============================================================
# fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily.nc"
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/LHS_integrated_buoyancy_tendency/LHS_domain_avg_timeseries.nc"
LHS_true_avg = xr.open_dataset(fname).LHS_true_avg



# ==============================================================
# 2. Vertical flux contribution
# ==============================================================
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day_Bflux.nc"
Bflux_daily_avg = xr.open_dataset(fname).Bflux_daily_avg
Bflux_daily_avg = -Bflux_daily_avg
Bflux_daily_avg = Bflux_daily_avg.assign_coords(time=LHS_true_avg.time.dt.floor("D"))

# vert = -Bflux_daily_avg * rho0/g/delta_rho * 86400
vert =  Bflux_daily_avg

# ==============================================================
# 4. wb
# ==============================================================
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_daily_1_16deg/wb_mld_horizontal_timeseries.nc"
wb_eddy_mean = xr.open_dataset(fname).wb_eddy_mean

wb_eddy =  const * wb_eddy_mean 


# ==============================================================
# 4. Steric |∇η′|
# ==============================================================
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly_timeseries/grad2_timeseries.nc"
eta_prime_grad2_mean = xr.open_dataset(fname).eta_prime_grad2_mean

hori_steric = Ce/abs_f * eta_prime_grad2_mean * (g**2)


# ==============================================================
# SSH submesoscale 
# ==============================================================
fname14 = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_Gaussian_submeso_14kmCutoff_timeseries.nc"
eta_submeso_grad2_14 = xr.open_dataset(fname14).eta_submeso_grad2_mean

hori_submeso_14 = Ce/abs_f * eta_submeso_grad2_14 * (g**2)

# ==============================================================
# 5. Ekman buoyancy flux
# ==============================================================
# fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux/B_Ek_timeseries.nc"
# B_Ek_mean = xr.open_dataset(fname).B_Ek_mean.isel(time=slice(1, None))
# B_Ek_mean = B_Ek_mean.assign_coords(time=B_Ek_mean.time.dt.floor("D"))
# tendency_ekman = -B_Ek_mean * rho0/g/delta_rho * 86400

fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux/B_Ek_domain_mean_timeseries.nc"
B_Ek_mean_hourly = xr.open_dataset(fname).B_Ek_mean
B_Ek_mean = B_Ek_mean_hourly.resample(time="1D").mean() # Compute daily mean
tendency_ekman = - B_Ek_mean 


# diff = dHml_dt - vert - tendency_ekman

residual = LHS_true_avg - vert - tendency_ekman - wb_eddy

# ==============================================================
# 7. Cumulative integrals
# ==============================================================
dt = 1.0

def cumulative(ds):
    return (ds.isel(time=slice(1, None)) * dt).cumsum(dim="time")

# Hml_total_cum = cumulative(dHml_dt)

#### when vert <= 0 , set vert = 0
# vert_clipped = vert.where((vert+tendency_ekman) > 0, 0)
# vert_cum = vert_clipped.cumsum(dim="time")
vert_cum = cumulative(vert)

#### when tendency_ekman <= 0 , set tendency_ekman = 0
# tendency_ekman_clipped = tendency_ekman.where((vert+tendency_ekman) > 0, 0)
# tendency_ek_cum = tendency_ekman_clipped.cumsum(dim="time")
tendency_ek_cum = cumulative(tendency_ekman)

# diff_cum = cumulative(diff)

hori_steric_cum = cumulative(hori_steric)
hori_submeso_cum_14 = cumulative(hori_submeso_14)

wb_eddy_cum = cumulative(wb_eddy)


# ==============================================================
# 6. Plots
# ==============================================================

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Hml_tendency/"
os.makedirs(figdir, exist_ok=True)
darkpink = '#d1007c'


# ==============================================================
# Plot 1 — Hml tendency
# ==============================================================
filename = f"{figdir}LHS_true.png"
plt.figure(figsize=(12, 6.5))

plt.plot(LHS_true_avg.time, LHS_true_avg, label="LHS_true_avg", color='k')
plt.plot(vert.time, vert, label="Surface buoyancy flux", color='tab:blue')
plt.plot(tendency_ekman.time, tendency_ekman, label="Ekman buoyancy flux", color='cyan')
# plt.plot(diff.time, diff, label="dHml/dt - surf - Ekman", linestyle='--', color='tab:green')

plt.plot(hori_steric.time, hori_steric, label="steric", color=darkpink)
plt.plot(hori_submeso_14.time, hori_submeso_14, label="SSH submeso 14 km", color='orange', linestyle='-')
plt.plot(wb_eddy.time, wb_eddy, label=r"$B_{eddy}$", color='purple', linestyle='-')
# plt.plot(
#     wb_eddy.time,
#     wb_eddy,
#     label=(
#         r"$\rho_0 \,/\, g \,/\, \Delta\rho \,"
#         r"\left( \langle|\overline{wb}^z|\rangle"
#         r" - \langle|\overline{w}^z \, \overline{b}^z|\rangle \right)$"
#     ),
#     color="purple",
#     linestyle="-"
# )
plt.plot(residual.time, residual, label="Residual", color='gray', linestyle='-')


plt.title("Mixed Layer Depth Tendency")
plt.ylabel("Rate of change of MLD [m/day]")
plt.xlabel("Time")
plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
plt.tight_layout()
plt.savefig(filename, dpi=200, bbox_inches='tight')



# ==============================================================
# 7-day rolling mean — Hml tendency
# ==============================================================

# 7-day rolling mean
window = 7

LHS_rm = LHS_true_avg.rolling(time=window, center=True).mean()
vert_rm = vert.rolling(time=window, center=True).mean()
tendency_ekman_rm = tendency_ekman.rolling(time=window, center=True).mean()
# diff_rm = diff.rolling(time=window, center=True).mean()
hori_steric_rm = hori_steric.rolling(time=window, center=True).mean()
hori_submeso_14_rm = hori_submeso_14.rolling(time=window, center=True).mean()
wb_eddy_rm = wb_eddy.rolling(time=window, center=True).mean()
residual_rm = residual.rolling(time=window, center=True).mean()


# ==============================================================
# Plot — 7-day rolling mean tendency
# ==============================================================

filename = f"{figdir}LHS_true_rolling_mean.png"
plt.figure(figsize=(12, 6.5))

plt.plot(LHS_rm.time, LHS_rm,
         label="LHS", color='k')

plt.plot(vert_rm.time, vert_rm,
         label="Surface buoyancy flux", color='tab:blue')

plt.plot(tendency_ekman_rm.time, tendency_ekman_rm,
         label="Ekman buoyancy flux", color='cyan')

# plt.plot(diff_rm.time, diff_rm,
#          label="dHml/dt - surf - Ekman", linestyle='--', color='tab:green')

plt.plot(hori_steric_rm.time, hori_steric_rm,
         label="steric", color=darkpink)

plt.plot(hori_submeso_14_rm.time, hori_submeso_14_rm,
         label="SSH submeso 14 km", color='orange')

plt.plot(wb_eddy_rm.time, wb_eddy_rm,
         label=r"$B_{eddy}$", color='purple')

# Residual — thicker line, lighter gray
plt.plot(residual_rm.time, residual_rm,
         label="Residual",
         color='lightgray',
         linewidth=2)


plt.title("Mixed Layer Depth Tendency (7-day rolling mean)")
plt.ylabel("Rate of change of MLD [m/day]")
plt.xlabel("Time")
plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
plt.tight_layout()
plt.savefig(filename, dpi=200, bbox_inches='tight')




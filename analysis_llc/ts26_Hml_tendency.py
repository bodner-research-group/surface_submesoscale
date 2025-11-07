##### Decompose the change in mixed layer depth (or vertical stratification as in Johnson et al. (2020a)) into 
##### horizontal processes (frontal slumping by eddies) and 
##### vertical processes (ocean surface buoyancy fluxes, turbulent mixing, vertical advection).

import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# from set_constant import domain_name, face, i, j

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

##### Load LLC dataset
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lat = ds1['YC'].isel(face=face, i=i, j=j)
Coriolis = 4*np.pi/86164*np.sin(lat*np.pi/180)
abs_f = np.abs(Coriolis.mean(dim=("i","j")).values) ### Absolute value of the coriolis parameter averaged over this domain

##### Define constants 
g = 9.81
rho0 = 1025
delta_rho = 0.03 ### The threshold for computing mixed-layer depth
Ce = 0.08


##### 1. Compute the tendency of Hml from Hml timeseries
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily.nc" # Hml_weekly_mean.nc
Hml_mean = abs(xr.open_dataset(fname).Hml_mean)
dHml_dt = Hml_mean.differentiate(coord="time")*1e9*86400  # dHml/dt, central difference, convert unit from m/ns to m/s and to m/day
dHml_dt= dHml_dt.assign_coords(time=dHml_dt.time.dt.floor("D"))


##### 2. The change in Hml due to net surface buoyancy flux (vertical process)
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day_Bflux.nc"
Bflux_daily_avg = xr.open_dataset(fname).Bflux_daily_avg
Bflux_daily_avg = - Bflux_daily_avg
Bflux_daily_avg= Bflux_daily_avg.assign_coords(time=dHml_dt.time.dt.floor("D"))
# vert = -Bflux_daily_avg*rho0/g/delta_rho *86400 # unit: m/day
vert = -Bflux_daily_avg*rho0/g/delta_rho*(Hml_mean-10)**2/(Hml_mean**2) *86400 # unit: m/day

diff = dHml_dt - vert


##### 3. The change in Hml due to mixed-layer eddy-induced frontal slumping (horizontal process)
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily.nc"
M2_mean = xr.open_dataset(fname).M2_mean  
# sigma_avg = 208/21 ### sigma_avg~9.9
sigma_avg = 1
# hori = -sigma_avg*Ce/abs_f*(M2_mean**2)*rho0/g/delta_rho *86400 * Hml_mean**2  # unit: m/day
hori = -sigma_avg*Ce/abs_f*(M2_mean**2)*rho0/g/delta_rho *86400 * (Hml_mean-10)**2  # unit: m/day

# grad_mag_rho_squared = (M2_mean*rho0/g)**2
# hori = 104/21*Ce/abs_f/delta_rho * grad_mag_rho_squared*86400 # unit: m/day

##### 4. Connect the gradient magnitude of mixed-layer steric height to mixed-layer horizontal density gradient

fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly_timeseries/grad2_timeseries.nc"
eta_grad2_mean = xr.open_dataset(fname).eta_grad2_mean.isel(time=slice(1, None))
eta_prime_grad2_mean = xr.open_dataset(fname).eta_prime_grad2_mean.isel(time=slice(1, None)) 

hori_steric = -sigma_avg*Ce/abs_f* eta_prime_grad2_mean *g*rho0/delta_rho *86400 * (Hml_mean-10)**2/(Hml_mean**2)  # unit: m/day

fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly_timeseries/grad2_submeso_timeseries.nc"
eta_submeso_grad2_mean = xr.open_dataset(fname).eta_submeso_grad2_mean.isel(time=slice(1, None)) 

hori_submeso = -sigma_avg*Ce/abs_f* eta_submeso_grad2_mean *g*rho0/delta_rho *86400 * (Hml_mean-10)**2/(Hml_mean**2)  # unit: m/day


##### 5. Apply a 7-day rolling mean

dHml_dt_rolling = dHml_dt.rolling(time=7, center=True).mean()
vert_rolling = vert.rolling(time=7, center=True).mean()
hori_rolling = hori.rolling(time=7, center=True).mean()
hori_steric_rolling = hori_steric.rolling(time=7, center=True).mean()
hori_submeso_rolling = hori_submeso.rolling(time=7, center=True).mean()

diff_rolling = dHml_dt_rolling - vert_rolling


# ==============================================================
# 6. Plot comparison
# ==============================================================

### Path to the folder where figures will be saved 
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Hml_tendency/"
os.makedirs(figdir, exist_ok=True)

filename=f"{figdir}Hml_tendency_new.png"
plt.figure(figsize=(15, 8))
plt.plot(dHml_dt["time"], dHml_dt, label=r"$dH_{ml}/dt$ (total)", color='k')
plt.plot(vert["time"], vert, label=r"Vertical process (surface buoyancy flux)", color='tab:blue')
plt.plot(diff["time"], diff, linestyle='--',label=r"$dH_{ml}/dt$-vertical", color='tab:green')
plt.plot(hori["time"], hori, label=r"Hori. process (eddy-induced frontal slumping)", color='tab:orange')
plt.plot(hori_steric["time"], hori_steric/4, label=r"Hori. (using steric |∇η′|)", color='yellow')
plt.plot(hori_submeso["time"], hori_submeso, label=r"Hori. (using submeso |∇η′|)", color='red')
plt.title("Mixed Layer Depth Tendency")
plt.ylabel("Rate of change of MLD [m/day]")
plt.xlabel("Time")
plt.legend(loc='lower right',bbox_to_anchor=(1.1, 0.02),borderaxespad=0)
plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
plt.savefig(filename, dpi=200)


filename=f"{figdir}Hml_tendency_new_rolling.png"
plt.figure(figsize=(15, 8))
plt.plot(dHml_dt_rolling["time"], dHml_dt_rolling, label=r"$dH_{ml}/dt$ (total)", color='k')
plt.plot(vert_rolling["time"], vert_rolling, label=r"Vertical process (surface buoyancy flux)", color='tab:blue')
plt.plot(diff_rolling["time"], diff_rolling, linestyle='--',label=r"$dH_{ml}/dt$-vertical", color='tab:green')
plt.plot(hori_rolling["time"], hori_rolling, label=r"Horizontal process (eddy-induced frontal slumping)", color='tab:orange')
plt.plot(hori_steric_rolling["time"], hori_steric_rolling/4, label=r"Horizontal (using steric |∇η′|)", color='yellow')
plt.plot(hori_submeso_rolling["time"], hori_submeso_rolling, label=r"Horizontal (using submeso |∇η′|)", color='red')
plt.title("Mixed Layer Depth Tendency (7-day rolling mean)")
plt.ylabel("Rate of change of MLD [m/day]")
plt.xlabel("Time")
plt.legend(loc='lower right',bbox_to_anchor=(1.1, 0.02),borderaxespad=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(filename, dpi=200)
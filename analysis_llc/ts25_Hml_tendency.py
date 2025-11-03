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
abs_f = Coriolis.mean(dim=("i","j")).values ### Absolute value of the coriolis parameter averaged over this domain

##### Define constants 
g = 9.81
rho0 = 1025
delta_rho = 0.03 ### The threshold for computing mixed-layer depth
Ce = 0.08


##### 1. Compute the tendency of Hml from Hml timeseries
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_7d_rolling.nc" # Hml_weekly_mean.nc
Hml_mean = abs(xr.open_dataset(fname).Hml_mean)
dHml_dt = Hml_mean.differentiate(coord="time")*1e9*86400  # dHml/dt, central difference, convert unit from m/ns to m/s and to m/day

##### 2. The change in Hml due to net surface buoyancy flux (vertical process)
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day_Bflux.nc"
Bflux_7day_smooth = xr.open_dataset(fname).Bflux_7day_smooth
Bflux_7day_smooth = - Bflux_7day_smooth
vert = -Bflux_7day_smooth*rho0/g/delta_rho *86400 # unit: m/day

##### 3. The change in Hml due to mixed-layer eddy-induced frontal slumping (horizontal process)
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_7d_rolling.nc"
M2_mean = xr.open_dataset(fname).M2_mean  
grad_mag_rho_squared = M2_mean*rho0/g

hori = 104/21*Ce/abs_f/delta_rho * grad_mag_rho_squared

##### 4. Connect the gradient magnitude of mixed-layer steric height to mixed-layer horizontal density gradient




# ==============================================================
# 5. Plot comparison
# ==============================================================

### Path to the folder where figures will be saved 
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Hml_tendency/"
os.makedirs(figdir, exist_ok=True)

filename=f"{figdir}Hml_tendency_test.png"
plt.figure(figsize=(10, 5))
plt.plot(dHml_dt["time"], dHml_dt, label=r"$dH_{ml}/dt$ (total)", color='k')
plt.plot(vert["time"], vert, label=r"Vertical process ($B_{flux}$)", color='tab:blue')
plt.title("Mixed Layer Depth Tendency and Vertical Forcing")
plt.ylabel("Rate of change of MLD [m/day]")
plt.xlabel("Time")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(filename, dpi=150)
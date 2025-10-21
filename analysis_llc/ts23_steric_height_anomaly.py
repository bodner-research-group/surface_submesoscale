#### Compute the steric height anomaly, following ECCO V4 Python Tutorial created by Andrew Delman:
#### https://ecco-v4-python-tutorial.readthedocs.io/Steric_height.html


# ===== Imports =====
import os
import numpy as np
import xarray as xr
import gsw
from datetime import timedelta
from dask.distributed import Client, LocalCluster
from set_constant import domain_name, face, i, j

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly"
os.makedirs(output_dir, exist_ok=True)

# ========== Open LLC4320 dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# ========== Extract subregion ==========
lon = ds1['XC'].isel(face=face, i=i, j=j)
lat = ds1['YC'].isel(face=face, i=i, j=j)
depth = ds1.Z                    # 1D vertical coordinate
drF = ds1.drF  # vertical grid spacing, 1D
drF3d, _, _ = xr.broadcast(drF, lon, lat)

# ========== Open 7-day rolling mean of T, S, potential density, alpha, and beta ==========
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_7d_rolling_mean"

# ========== Define constants ==========
g = 9.81
rhoConst = 1029.
p_atm = 101325./1e4   # atmospheric pressure at sea surface, in dbar

# ========== Estimate hydrostatic Pressure ==========
# Pressure in dbar from depth (positive down), add atmospheric pressure (converted to dbar)
pressure = gsw.p_from_z(np.abs(depth), lat)  # in dbar

# ========== Load SSH ==========
# print("Loading daily averaged Eta...")
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds_eta = xr.open_mfdataset(eta_path, combine='by_coords')

# Daily averaged SSH
Eta_daily = ds_eta["Eta"] # Align datasets and select face/i/j region

# Compute 7-day rolling mean of SSH
Eta = Eta_daily.rolling(time=7, center=True).mean() 

# ===================================================
# ========== Compute steric height anomaly ==========
# ===================================================

eta = Eta.isel(time=10)
eta.time.values

input_file = os.path.join(input_dir, "rho_Hml_TS_7d_20111111.nc")
date_tag = os.path.basename(input_file).split("_")[-1].replace(".nc", "")
ds = xr.open_dataset(input_file, chunks={"k": 10, "j": 100, "i": 100})

# ========== Compute in-situ densiy and hydrostatic pressure ==========
theta = ds.T_7d
salt = ds.S_7d

# Prepare empty arrays
SA = xr.zeros_like(salt)
CT = xr.zeros_like(theta)
rho_insitu = xr.zeros_like(theta)
pres_hydro = xr.zeros_like(theta)  ### hydrostatic pressure (absolute pressure minus 10.1325 dbar), dbar
P0 = 0  # Surface pressure minus p_atm, in dbar

# Get vertical dimension name and size
kdim = "k"  # change if your vertical dimension has a different name
Nr = len(theta[kdim])

# Loop over vertical levels
### LLC4320 uses cell-centered approach: https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html
for k in range(Nr):
    z = np.abs(depth.isel({kdim: k}))  # [m], 2D (y, x)
    sp = salt.isel({kdim: k})
    th = theta.isel({kdim: k})

    if k == 0:
        pres_k_estimated = z
        SA_k = gsw.SA_from_SP(sp, pres_k_estimated, lon, lat)
        CT_k = gsw.CT_from_pt(SA_k, th)
        rho_k = gsw.rho(SA_k, CT_k, pres_k_estimated)
        pres_k = P0 + g * rho_k * 0.5*drF.isel({kdim: k}) / 1e4  # [dbar]
    else:
        rho_prev = rho_insitu.isel({kdim: k-1})
        pres_prev = pres_hydro.isel({kdim: k-1})
        #### Use rho_prev to approximate rho_k when estimating pressure at level k
        pres_k_estimated = pres_prev + g * (rho_prev* 0.5*drF.isel({kdim: k-1}) + rho_prev* 0.5*drF.isel({kdim: k})) / 1e4 
        SA_k = gsw.SA_from_SP(sp, pres_k_estimated, lon, lat)
        CT_k = gsw.CT_from_pt(SA_k, th)
        rho_k = gsw.rho(SA_k, CT_k, pres_k_estimated)
        pres_k = pres_prev + g * (rho_prev* 0.5*drF.isel({kdim: k-1}) + rho_k* 0.5*drF.isel({kdim: k})) / 1e4 

    # Assign back to xarray objects
    SA.loc[{kdim: k}] = SA_k
    CT.loc[{kdim: k}] = CT_k
    rho_insitu.loc[{kdim: k}] = rho_k
    pres_hydro.loc[{kdim: k}] = pres_k

# ========== Specific volume anomaly ==========
# compute standard specific volume and anomalies
S_Ar = 35.16504    # absolute salinity standard for spec. vol., notated as SSO in GSW documentation
T_Cr = 0.          # conservative temperature standard
specvol_standard = gsw.density.specvol(S_Ar,T_Cr,pres_hydro.values)

specvol_anom = 1/rho_insitu - specvol_standard

# ========== Steric height anomaly ==========
# pressure reference level to compute steric height
# (in units of dbar, minus 10.1325 dbar atmospheric pressure)
p_top_sea_dbar = 0.
p_top = (p_top_sea_dbar) + p_atm ### dbar
p_r_sea_dbar = 1000.
p_r = (p_r_sea_dbar) + p_atm     ### dbar

# compute pressure at z = 0 (not exactly the ocean surface)
press_z0 = p_atm + g*rho_insitu.isel(k=0)*eta /1e4

# integrate hydrostatic balance downward to get pressure at bottom of grid cells
press_ku = press_z0 + (rho_insitu*g*ds1.drF).cumsum("k")
# press_ku.Z.values = ds1.Zu.values

# create array with pressure at top of grid cells
press_kl = xr.concat([press_z0,press_ku.isel(k=np.arange(len(ds1.k) - 1))],dim="k")
press_kl = press_kl.assign_coords(k=ds1.k.values)

# compute dp for this integration
dp_integrate =  np.fmax(press_kl,p_top*np.ones(press_kl.shape)) - \
                np.fmin(press_ku,p_r*np.ones(press_ku.shape))        ### dbar 

# Integrate specific volume anomaly over depth
steric_height_anom = (-(specvol_anom/g)*dp_integrate*1e4).sum("k") ### in meters

# ========== Compare steric height with sea surface height ==========
ssh_diff = eta - steric_height_anom

# ===============================================================
# ========== Thermosteric and halosteric contributions ==========
# ===============================================================

# Thermosteric: keep salinity constant at mean value
salt_mean = salt.mean(dim='k')
SA_mean = gsw.SA_from_SP(salt_mean, pres_hydro, lon, lat)
CT_temp = gsw.CT_from_pt(SA_mean, theta)
rho_temp = gsw.rho(SA_mean, CT_temp, pres_hydro)
specvol_anom_temp = (1 / rho_temp) - (1 / rhoConst)
thermosteric_height = (specvol_anom_temp * drF3d).sum(dim='k')

# Halosteric: keep temperature constant at mean value
theta_mean = theta.mean(dim='k')
CT_salt = gsw.CT_from_pt(SA, theta_mean)
rho_salt = gsw.rho(SA, CT_salt, pres_hydro)
specvol_anom_salt = (1 / rho_salt) - (1 / rhoConst)
halosteric_height = (specvol_anom_salt * drF3d).sum(dim='k')

# ========== Save output ==========
out_ds = xr.Dataset(
    {
        "steric_height_anomaly": steric_height_anom,
        "thermosteric_height": thermosteric_height,
        "halosteric_height": halosteric_height,
    },
    coords={
        "i": ds.i,
        "j": ds.j,
        "face": face,
        "XC": lon,
        "YC": lat,
    },
    attrs={
        "description": "Steric height anomaly and thermosteric/halosteric contributions",
        "rhoConst": rhoConst,
        "date": date_tag,
    }
)

output_file = os.path.join(output_dir, f"steric_height_anomaly_{date_tag}.nc")
out_ds.to_netcdf(output_file)

print(f"💡 Saved steric height anomaly data to {output_file}")



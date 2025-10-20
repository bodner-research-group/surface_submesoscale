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

# ========== Open 7-day rolling mean of T, S, potential density, alpha, and beta ==========
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_7d_rolling_mean"

# ========== Extract subregion ==========
lon = ds1['XC'].isel(face=face, i=i, j=j)
lat = ds1['YC'].isel(face=face, i=i, j=j)
depth = ds1.Z                    # 1D vertical coordinate
dz = ds1.drF  # vertical grid spacing, 1D
dz3d, _, _ = xr.broadcast(dz, lon, lat)

# ========== Define constants ==========
g = 9.81
rhoConst = 1029.
p_atm = 101325.   # atmospheric pressure at sea surface, in Pa

# ========== Estimate hydrostatic Pressure ==========
# Pressure in dbar from depth (positive down), add atmospheric pressure (converted to dbar)
pressure = gsw.p_from_z(depth, lat)  # in dbar


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

input_file = os.path.join(input_dir, "rho_Hml_TS_7d_20111104.nc")
date_tag = os.path.basename(input_file).split("_")[-1].replace(".nc", "")
ds = xr.open_dataset(input_file, chunks={"k": 10, "j": 100, "i": 100})

# ========== Compute in-situ densiy ==========
theta = ds.T_7d
salt = ds.S_7d
SA = gsw.SA_from_SP(salt, pressure, lon, lat)
CT = gsw.CT_from_pt(SA, theta)
rho_insitu = gsw.rho (SA, CT, pressure)

# ========== Specific volume anomaly ==========
# Anomaly is defined relative to rhoConst
# specvol_anom = (1 / rho_insitu) - (1 / rhoConst)

# compute standard specific volume and anomalies
S_Ar = 35.16504    # absolute salinity standard for spec. vol., notated as SSO in GSW documentation
T_Cr = 0.    # conservative temperature standard
specvol_standard = gsw.density.specvol(S_Ar,T_Cr,pressure)

specvol_anom = 1/rho_insitu - specvol_standard

# ========== Steric height anomaly ==========
# Integrate specific volume anomaly over depth
steric_height_anom = (specvol_anom * dz3d).sum(dim='k')

# Convert to meters
# Already in meters due to dz being in meters and specvol_anom in 1/kg
# But often we divide by gravity if using pressure, which we’re not here
# So no need to divide by g

# ========== Compare steric height with sea surface height ==========
eta = Eta.isel(time=0)
ssh_diff = eta - steric_height_anom


# ===============================================================
# ========== Thermosteric and halosteric contributions ==========
# ===============================================================

# Thermosteric: keep salinity constant at mean value
salt_mean = salt.mean(dim='k')
SA_mean = gsw.SA_from_SP(salt_mean, pressure_hydro, lon, lat)
CT_temp = gsw.CT_from_pt(SA_mean, theta)
rho_temp = gsw.rho(SA_mean, CT_temp, pressure_hydro)
specvol_anom_temp = (1 / rho_temp) - (1 / rhoConst)
thermosteric_height = (specvol_anom_temp * dz3d).sum(dim='k')

# Halosteric: keep temperature constant at mean value
theta_mean = theta.mean(dim='k')
CT_salt = gsw.CT_from_pt(SA, theta_mean)
rho_salt = gsw.rho(SA, CT_salt, pressure_hydro)
specvol_anom_salt = (1 / rho_salt) - (1 / rhoConst)
halosteric_height = (specvol_anom_salt * dz3d).sum(dim='k')

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



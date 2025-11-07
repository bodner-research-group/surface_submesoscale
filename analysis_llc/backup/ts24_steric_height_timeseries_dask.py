#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute steric height anomaly gradients and Laplacians
for the Icelandic Basin domain using Dask parallelism,
and save results in NetCDF format.
"""

import os
import numpy as np
import xarray as xr
import gsw
from glob import glob
import matplotlib.pyplot as plt
from xgcm import Grid
import pandas as pd
from dask.distributed import Client, LocalCluster
from dask import compute

# ==============================================================
# Domain setup
# ==============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ==============================================================
# Dask setup
# ==============================================================
cluster = LocalCluster(n_workers=4, threads_per_worker=16, memory_limit="90GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ==============================================================
# Parameters and paths
# ==============================================================
g = 9.81
rhoConst = 1029.0
p_atm = 101325.0 / 1e4  # dbar

base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
eta_dir = os.path.join(base_dir, "surface_24h_avg")
rho_dir = os.path.join(base_dir, "rho_insitu_hydrostatic_pressure_daily")
Hml_file = os.path.join(base_dir, "Lambda_MLI_timeseries_daily.nc")

out_dir = os.path.join(base_dir, "steric_height_anomaly_timeseries")
os.makedirs(out_dir, exist_ok=True)

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/steric_height/"
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Load grid
# ==============================================================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lon = ds1["XC"].isel(face=face, i=i, j=j)
lat = ds1["YC"].isel(face=face, i=i, j=j)
depth = ds1.Z
drF = ds1.drF
drF3d, _, _ = xr.broadcast(drF, lon, lat)

ds_grid_face = ds1.isel(face=face, i=i, j=j, i_g=i, j_g=j, k=0, k_p1=0, k_u=0)
if "time" in ds_grid_face.dims:
    ds_grid_face = ds_grid_face.isel(time=0, drop=True)
ds1.close()

coords = {"X": {"center": "i", "left": "i_g"}, "Y": {"center": "j", "left": "j_g"}}
metrics = {("X",): ["dxC", "dxG"], ("Y",): ["dyC", "dyG"]}
grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)

# ==============================================================
# Load data with Dask parallelism
# ==============================================================
print("Loading Eta dataset...")
ds_eta = xr.open_mfdataset(
    os.path.join(eta_dir, "eta_24h_*.nc"),
    combine="by_coords",
    chunks={"time": 1, "i": -1, "j": -1},
)
Eta = ds_eta["Eta"].assign_coords(time=ds_eta.time.dt.floor("D"))
ds_eta.close()

print("Loading rho_insitu dataset (manual time coordinate)...")
rho_files = sorted(glob(os.path.join(rho_dir, "rho_insitu_pres_hydro_*.nc")))
def get_date_from_filename(fname):
    # e.g. rho_insitu_pres_hydro_20130415.nc → 2013-04-15
    date_str = os.path.basename(fname).split("_")[-1].split(".")[0]
    return np.datetime64(pd.to_datetime(date_str, format="%Y%m%d"))
# Open each file and manually assign a time dimension
datasets = []
for f in rho_files:
    ds_tmp = xr.open_dataset(f, chunks={"k": -1, "j": -1, "i": -1})
    ds_tmp = ds_tmp.expand_dims(time=[get_date_from_filename(f)])
    datasets.append(ds_tmp)

# Combine along new time dimension
ds_rho = xr.concat(datasets, dim="time")
rho_insitu = ds_rho["rho_insitu"]
ds_rho.close()

print("Loading Hml dataset...")
Hml_mean = xr.open_dataset(Hml_file).Hml_mean.assign_coords(
    time=lambda x: x.time.dt.floor("D")
)


# ==============================================================
# Helper functions
# ==============================================================
def compute_grad_laplace(var, grid):
    """Compute gradient magnitude and Laplacian using xgcm."""
    var_x = grid.derivative(var, axis="X")
    var_y = grid.derivative(var, axis="Y")
    var_x_c = grid.interp(var_x, axis="X", to="center")
    var_y_c = grid.interp(var_y, axis="Y", to="center")
    grad_mag = np.sqrt(var_x_c**2 + var_y_c**2)
    var_xx = grid.derivative(var_x_c, axis="X")
    var_yy = grid.derivative(var_y_c, axis="Y")
    var_xx_c = grid.interp(var_xx, axis="X", to="center")
    var_yy_c = grid.interp(var_yy, axis="Y", to="center")
    laplace = var_xx_c + var_yy_c
    return grad_mag, laplace


# ==============================================================
# Compute anomalies (fully lazy)
# ==============================================================
print("Computing rho′ and η′ (lazy)...")

rho_prime = rho_insitu - rho_insitu.mean(dim=("i", "j"))

# Interpolate Hml_mean to match Eta time
Hml_interp = Hml_mean.interp(time=Eta.time)

mask3d = (depth >= Hml_interp).broadcast_like(rho_prime)
rho_prime_masked = rho_prime.where(mask3d)
drF_masked = drF3d.where(mask3d)

eta_prime = -(1 / rhoConst) * (rho_prime_masked * drF_masked).sum(dim="k")
eta_prime = eta_prime - eta_prime.mean(dim=["i", "j"])

eta_mean = Eta - Eta.mean(dim=["i", "j"])

# ==============================================================
# Compute gradients and Laplacians
# ==============================================================
print("Computing gradients and Laplacians (Dask-lazy)...")

eta_grad_mag, eta_laplace = compute_grad_laplace(eta_mean, grid)
eta_prime_grad_mag, eta_prime_laplace = compute_grad_laplace(eta_prime, grid)

# ==============================================================
# Domain-mean quantities
# ==============================================================
eta_grad2_mean = (eta_grad_mag**2).mean(dim=["i", "j"])
eta_prime_grad2_mean = (eta_prime_grad_mag**2).mean(dim=["i", "j"])

# ==============================================================
# Combine results into dataset
# ==============================================================
print("Preparing output dataset...")

ds_out = xr.Dataset(
    {
        "eta": Eta,
        "eta_grad_mag": eta_grad_mag,
        "eta_laplace": eta_laplace,
        "eta_prime": eta_prime,
        "eta_prime_grad_mag": eta_prime_grad_mag,
        "eta_prime_laplace": eta_prime_laplace,
        "eta_grad2_mean": eta_grad2_mean,
        "eta_prime_grad2_mean": eta_prime_grad2_mean,
    },
    coords={"lon": lon, "lat": lat, "time": Eta.time},
)

# ==============================================================
# Save with Dask-delayed parallel computation
# ==============================================================
outfile = os.path.join(out_dir, "eta_steric_grad_laplace_daily.nc")
print(f"Saving dataset to NetCDF (Dask-delayed): {outfile}")
delayed_write = ds_out.to_netcdf(outfile, compute=True)
compute(delayed_write)
print("✅ Saved main dataset")

# ==============================================================
# Save domain-mean time series separately
# ==============================================================
ts_ds = xr.Dataset(
    {
        "eta_grad2_mean": eta_grad2_mean,
        "eta_prime_grad2_mean": eta_prime_grad2_mean,
    }
)
ts_path = os.path.join(out_dir, "grad2_timeseries.nc")
delayed_ts = ts_ds.to_netcdf(ts_path, compute=True)
compute(delayed_ts)
print(f"✅ Saved domain-mean timeseries: {ts_path}")

# ==============================================================
# Plot timeseries
# ==============================================================
plt.figure(figsize=(8, 4))
ts_ds["eta_grad2_mean"].plot(label="⟨|∇η|²⟩", color="tab:blue")
ts_ds["eta_prime_grad2_mean"].plot(label="⟨|∇η′|²⟩", color="tab:orange")

# 7-day rolling mean
ts_ds["eta_grad2_mean"].rolling(time=7, center=True).mean().plot(
    label="⟨|∇η|²⟩ (7d)", color="tab:blue", linestyle="--"
)
ts_ds["eta_prime_grad2_mean"].rolling(time=7, center=True).mean().plot(
    label="⟨|∇η′|²⟩ (7d)", color="tab:orange", linestyle="--"
)

plt.title("Domain-mean |∇η|² and |∇η′|² Timeseries")
plt.ylabel("Mean(|∇η|²) [m²/m²]")
plt.xlabel("Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{figdir}grad2_timeseries.png", dpi=150)
plt.close()
print(f"✅ Saved figure: {figdir}grad2_timeseries.png")

print("✅ All computations and NetCDF outputs completed successfully.")

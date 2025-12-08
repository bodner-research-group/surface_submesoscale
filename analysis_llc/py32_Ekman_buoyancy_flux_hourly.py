"""
Compute Ekman Buoyancy Flux (B_Ek) from LLC4320 hourly surf data
Memory-safe: process one day at a time
Output: 1 NetCDF per day
"""

import os
import numpy as np
import xarray as xr
import gsw
from dask.distributed import Client, LocalCluster
from xgcm import Grid

# ==================================================
# DASK CONFIG
# ==================================================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ==================================================
# Domain settings
# ==================================================
from set_constant import domain_name, face, i, j, start_hours, end_hours

# ==================================================
# Parameters
# ==================================================
rho0 = 1027.5
gravity = 9.81

# ==================================================
# Paths
# ==================================================
in_zarr = "/orcd/data/abodner/003/LLC4320/LLC4320"
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(out_dir, exist_ok=True)

# ==================================================
# Load grid (static)
# ==================================================
ds = xr.open_zarr(in_zarr, consolidated=False)

lon = ds.XC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
lat = ds.YC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})

Coriolis = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)

# Setup xgcm grid
# i_g = slice(i.start, i.stop + 1)
# j_g = slice(j.start, j.stop + 1)
i_g = i
j_g = j
ds_grid = ds.isel(face=face, i=i, j=j, i_g=i_g, j_g=j_g, k=0, k_p1=0, k_u=0).isel(time=0, drop=True)

coords = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
}
metrics = {
    ("X",): ["dxC", "dxG"],
    ("Y",): ["dyC", "dyG"],
}
grid = Grid(ds_grid, coords=coords, metrics=metrics, periodic=False)

# ==================================================
# Daily loop
# ==================================================
hours_per_day = 24
# for hour0 in range(start_hours, end_hours, hours_per_day):
for hour0 in range(end_hours - hours_per_day, start_hours - hours_per_day, -hours_per_day):

    # -------------------------
    # Get datetime of first timestep in this daily chunk and build filename
    # -------------------------
    t0 = ds.time.isel(time=hour0).values
    date_str = np.datetime_as_string(t0, unit="D")  # returns '2011-11-01'
    out_nc = f"{out_dir}/B_Ek_{date_str.replace('-', '')}.nc"

    if os.path.exists(out_nc):
        print(f"File exists, skipping: {out_nc}")
        continue

    hour1 = min(hour0 + hours_per_day, end_hours)
    print(f"\n===== PROCESSING HOURS {hour0} → {hour1} | DATE {date_str} =====")

    # --------------------------------------------------
    # Subset 24 hours
    # --------------------------------------------------
    salt = ds.Salt.isel(face=face, i=i, j=j, k=0,time=slice(hour0, hour1)).chunk({"time": 24, "j": -1, "i": -1})
    theta = ds.Theta.isel(face=face, i=i, j=j, k=0,time=slice(hour0, hour1)).chunk({"time": 24, "j": -1, "i": -1})

    taux = ds.oceTAUX.isel(face=face, j=j, i_g=i_g,time=slice(hour0, hour1)).chunk({"time": 24, "j": -1, "i_g": -1})
    tauy = ds.oceTAUY.isel(face=face, i=i, j_g=j_g,time=slice(hour0, hour1)).chunk({"time": 24, "j_g": -1, "i": -1})

    # ==================================================
    # Compute SA, CT, rho
    # ==================================================
    SA = xr.apply_ufunc(
        gsw.SA_from_SP,
        salt, 0, lon, lat,
        input_core_dims=[["j","i"], [], ["j","i"], ["j","i"]],
        output_core_dims=[["j","i"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
    )

    CT = xr.apply_ufunc(
        gsw.CT_from_pt,
        SA, theta,
        input_core_dims=[["j","i"], ["j","i"]],
        output_core_dims=[["j","i"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
    )

    rho = xr.apply_ufunc(
        gsw.rho,
        SA, CT, 0,
        input_core_dims=[["j","i"], ["j","i"], []],
        output_core_dims=[["j","i"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
    )

    buoy = -gravity * (rho - rho0) / rho0

    # ==================================================
    # Gradients
    # ==================================================
    dbdx = grid.interp(grid.derivative(buoy, axis="X"), axis="X", to="center")
    dbdy = grid.interp(grid.derivative(buoy, axis="Y"), axis="Y", to="center")

    # ==================================================
    # Wind stress to C-grid
    # ==================================================
    taux_c = grid.interp(taux, axis="X", to="center")
    tauy_c = grid.interp(tauy, axis="Y", to="center")

    # ==================================================
    # Ekman buoyancy flux
    # ==================================================
    B_Ek = (tauy_c * dbdx - taux_c * dbdy) / (rho0 * Coriolis)
    B_Ek = B_Ek.rename("B_Ek")
    B_Ek.attrs["units"] = "m^2/s^3"

    # ==================================================
    # Compute + Save
    # ==================================================
    print("Computing + writing daily file...")

    # Compute in Dask, write to disk
    B_Ek.compute().to_netcdf(out_nc)

    print(f"Saved: {out_nc}")

print("ALL DONE.")











##### ------------------------------------------------------------
##### SCRIPT 2 — COMPUTE DOMAIN-MEAN TIMESERIES FROM DAILY FILES
##### ------------------------------------------------------------

import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta

from set_constant import domain_name

in_dir  = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
out_nc  = f"{in_dir}/B_Ek_domain_mean_timeseries.nc"

files = sorted([f for f in os.listdir(in_dir) if f.startswith("B_Ek_20") and f.endswith(".nc")])

means = []
times = []

for fname in files:

    # --- get date from filename e.g. B_Ek_20120101.nc ---
    date_str = fname.split("_")[2].split(".")[0]   # '20120101'
    day0 = datetime.strptime(date_str, "%Y%m%d")

    # --- open dataset ---
    ds = xr.open_dataset(os.path.join(in_dir, fname))
    B = ds["B_Ek"]  # shape (24, j, i) but no time coords

    # --- create time array for the 24 hours ---
    t24 = np.array([day0 + timedelta(hours=h) for h in range(B.sizes["time"])])

    # --- assign new time axis ---
    B = B.assign_coords(time=("time", t24))

    # --- remove boundary ---
    Bsub = B.isel(i=slice(2, -2), j=slice(2, -2))

    # --- domain mean for the whole day (24 hours) ---
    Bmean_day = Bsub.mean(["i", "j"])

    # accumulate
    means.append(Bmean_day.values)
    times.append(t24)

# concatenate full time series
Bmean_ts = xr.DataArray(
    data=np.concatenate(means),
    coords={"time": np.concatenate(times)},
    dims="time",
    name="B_Ek_mean",
)
Bmean_ts.attrs["units"] = "m^2/s^3"

# save
Bmean_ts.to_netcdf(out_nc)
print("Saved:", out_nc)









##### ------------------------------------------------------------
##### SCRIPT 3 — PLOT FULL TIMESERIES + ROLLING MEAN
##### ------------------------------------------------------------

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from set_constant import domain_name
plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

in_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
ts_file = f"{in_dir}/B_Ek_domain_mean_timeseries.nc"

firdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Ekman_buoyancy_flux"

ds = xr.open_dataset(ts_file)
B = ds.B_Ek_mean

# 7-day rolling
roll = B.rolling(time=24*7, center=True).mean()

plt.figure(figsize=(10,5))
plt.plot(B.time, B, label="Domain mean hourly")
plt.plot(B.time, roll, label="7-day rolling mean", lw=2)
plt.legend()
plt.grid()
plt.title("Domain-mean Ekman Buoyancy Flux")
plt.xlabel("Time")
plt.ylabel("B_Ek (m²/s³)")
plt.tight_layout()

plt.savefig(f"{firdir}/B_Ek_timeseries.png", dpi=150)
print("Saved figure.")









##### ------------------------------------------------------------
##### SCRIPT 4 — Plot Maps of EBF from Jan 1 to Jan 10 (From NC Files)
##### ------------------------------------------------------------

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime, timedelta
from set_constant import domain_name, face, i, j, start_hours, end_hours

print("Plotting maps of Ekman buoyancy flux from Sept 1–10, 2012...")

# Coordinate
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lon = ds1.XC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
lat = ds1.YC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})

# Directories
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(figdir, exist_ok=True)

# Get files
nc_files = sorted(glob.glob(f"{out_dir}/B_Ek_201209*.nc"))

# keep only Jan 1–10
nc_files = [f for f in nc_files if "2012090" in f or "20120910" in f]  # safe filtering

if len(nc_files) == 0:
    raise RuntimeError("No .nc files found for Jan 1–10, 2012!")


# =====================================================
# Open and add correct time coordinates
# =====================================================
all_days = []
for fpath in nc_files:
    fname = os.path.basename(fpath)
    date_str = fname.split("_")[2].split(".")[0]  # '20120101'
    day0 = datetime.strptime(date_str, "%Y%m%d")

    ds = xr.open_dataset(fpath)
    # Create hourly time coordinate
    times = [day0 + timedelta(hours=h) for h in range(ds.dims["time"])]
    ds = ds.assign_coords(time=("time", times))

    all_days.append(ds)

# Concatenate along time
ds_all = xr.concat(all_days, dim="time")

# Extract B_Ek
B_Ek = ds_all["B_Ek"]

vmax = np.nanmax(np.abs(B_Ek))  # maximum absolute value across all data
vmin = -vmax                     # symmetric around zero

# =====================================================
# Plot maps for each hour
# =====================================================
for t in B_Ek.time.values:
    B_now = B_Ek.sel(time=t)

    t_str = np.datetime_as_string(t, unit='h')

    fig, ax = plt.subplots(figsize=(8, 10))

    max_Bnow = np.nanmax(np.abs(B_now))
    if max_Bnow == 0 or np.isnan(max_Bnow):
        print(f"Skipping time {t_str} because data is all zeros or NaN")
        continue

    im = ax.pcolormesh(lon, lat, B_now,
                    cmap="RdBu_r", shading="auto",
                    vmin=vmin/20, vmax=vmax/20)

    ax.set_title(f"Ekman Buoyancy Flux at {t_str}", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, label="B_Ek (m²/s³)")
    plt.savefig(f"{figdir}/EBF_map_{t_str}.png", dpi=150)
    plt.close()

print("DONE.")




# ============================================================
# Create animation
# ============================================================
print("Creating animation...")

##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Ekman_buoyancy_flux"
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Ekman_buoyancy_flux/movie-Ekman_buoyancy_flux.mp4"
os.system(f"ffmpeg -r 15 -pattern_type glob -i '{figdir}/EBF_map_20*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")


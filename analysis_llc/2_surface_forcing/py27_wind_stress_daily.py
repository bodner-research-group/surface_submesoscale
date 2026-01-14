import xarray as xr
import numpy as np
import os
import time
from dask.distributed import Client, LocalCluster
from xgcm import Grid
from tqdm.auto import tqdm

# =============================================================
# Constants / Domain
# =============================================================
from set_constant import start_hours, end_hours
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# =============================================================
# Paths
# =============================================================
# directories where the 24-hour avg files already exist
daily_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"

# output directory
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(output_dir, exist_ok=True)
outfile = os.path.join(output_dir, "windstress_center_daily_avg.nc")

# =============================================================
# Dask cluster
# =============================================================
cluster = LocalCluster(
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB"
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)


# =============================================================
# Helper: Load 24h averaged taux/tauy files
# =============================================================
def load_daily_series(label,variable):
    """Loads all daily files for taux or tauy and concatenates along time."""
    flist = sorted(
        [os.path.join(daily_dir, f) for f in os.listdir(daily_dir)
         if f.startswith(f"{label}_24h_") and f.endswith(".nc")]
    )
    if len(flist) == 0:
        raise FileNotFoundError(f"No daily files found for {label} in {daily_dir}")

    print(f"Loading {len(flist)} daily files for {label}...")

    ds = xr.open_mfdataset(flist, combine="by_coords", parallel=True)
    return ds[variable]

# =============================================================
# Load daily averages
# =============================================================
taux_daily = load_daily_series("taux","oceTAUX")
tauy_daily = load_daily_series("tauy","oceTAUY")


# =============================================================
# Open a small slice of the original LLC4320 dataset for grid metrics
# =============================================================
print("Loading LLC4320 grid metrics for interpolation...")
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# use ds1 for coordinates
ds_grid = ds1.isel(
    face=face,
    i=i, j=j,
    i_g=i, j_g=j,
    k=0, k_u=0, k_p1=0
)

if "time" in ds_grid.dims:
    ds_grid = ds_grid.isel(time=0, drop=True)

coords = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
}

metrics = {
    ("X",): ["dxC", "dxG"],
    ("Y",): ["dyC", "dyG"]
}

grid = Grid(ds_grid, coords=coords, metrics=metrics, periodic=False)

# =============================================================
# Interpolation to center grid
# =============================================================
print("\nInterpolating wind stress to the C-grid center...")

tstart = time.time()

taux_center = grid.interp(taux_daily, axis="X", to="center")
tauy_center = grid.interp(tauy_daily, axis="Y", to="center")

# Compute results
taux_center = taux_center.compute()
tauy_center = tauy_center.compute()

taux_center = taux_center.assign_coords(time=taux_daily.time)
tauy_center = tauy_center.assign_coords(time=tauy_daily.time)

print(f"Interpolation finished in {(time.time() - tstart)/60:.2f} min")

# =============================================================
# Save output
# =============================================================
print("\nSaving to NetCDF:", outfile)
ds_out = xr.Dataset({
    "taux_center": taux_center,
    "tauy_center": tauy_center
})

t0 = time.time()
ds_out.to_netcdf(outfile)
print(f"Saved in {(time.time() - t0)/60:.2f} min")

# Cleanup
client.close()
cluster.close()
print("\nDONE.")





# =============================================================
# Compute wind-stress magnitude and plot time series
# =============================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain


outfile = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux/windstress_center_daily_avg.nc"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wind_stress"
os.makedirs(figdir, exist_ok=True)

print("\nLoading saved center-interpolated wind stress...")
ds = xr.open_dataset(outfile)

taux = ds["taux_center"]
tauy = ds["tauy_center"]

# Wind-stress magnitude
print("Computing wind-stress magnitude...")
tau_mag = np.sqrt(taux**2 + tauy**2)

# Spatial average (if desired)
tau_mag_mean = tau_mag.mean(dim=("i", "j"))

# =============================================================
# Plot time series
# =============================================================
plt.figure(figsize=(12, 5))
tau_mag_mean.plot(color='k', linewidth=1.5)

plt.title("Daily Mean Wind-Stress Magnitude")
plt.xlabel("Time")
plt.ylabel("Wind Stress Magnitude [N/m²]")
plt.grid(True)

plot_file = os.path.join(figdir, "windstress_magnitude_timeseries.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.show()

print(f"Time series plot saved to: {plot_file}")
print("DONE plotting.")





# # =============================================================
# # Select surface wind stress (full hourly time window)
# # =============================================================
# taux_hr = ds1["oceTAUX"].isel(face=face, i_g=i, j=j).isel(time=slice(start_hours, end_hours)).chunk({"time": 24, "i_g": 20, "j": 20})
# tauy_hr = ds1["oceTAUY"].isel(face=face, i=i, j_g=j).isel(time=slice(start_hours, end_hours)).chunk({"time": 24, "i": 20, "j_g": 20})

# # =============================================================
# # DAILY AVERAGING FIRST
# # =============================================================
# nt = taux_hr.sizes["time"]
# n_days = nt // 24

# print(f"\nComputing {n_days} daily averages first...")

# taux_daily_list = []
# tauy_daily_list = []

# for d in tqdm(range(n_days)):
#     t0, t1 = d * 24, (d + 1) * 24
#     taux_daily_list.append(taux_hr.isel(time=slice(t0, t1)).mean("time").compute())
#     tauy_daily_list.append(tauy_hr.isel(time=slice(t0, t1)).mean("time").compute())

# # Combine into daily fields
# taux_daily = xr.concat(taux_daily_list, dim="time")
# tauy_daily = xr.concat(tauy_daily_list, dim="time")


# # taux_daily = taux_hr.coarsen(time=24, boundary='trim').mean().compute()
# # tauy_daily = tauy_hr.coarsen(time=24, boundary='trim').mean().compute()

### Compute and plot the mean stratification in 50%-90% of the ML

# ========== Imports ==========
import os
import numpy as np
import xarray as xr
from glob import glob
from dask.distributed import Client, LocalCluster

from set_constant import domain_name

# ========== Constants ==========
g = 9.81
rho0 = 1025

# ========== Setup Dask ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_weekly"
output_file = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/", "N2ml_weekly.nc")
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
files = sorted(glob(os.path.join(input_dir, "rho_Hml_TS_7d_*.nc")))
# files = sorted(glob(os.path.join(input_dir, "rho_Hml_TS_7d_2012-01-24.nc")))

# ========== Load grid ==========
print("Loading vertical grid from Zarr...")
ds_grid = xr.open_zarr(zarr_path, consolidated=False)
depth_1d = ds_grid.Z.values
ds_grid.close()

# ========== Compute N² function ==========
def compute_N2(rho, depth):
    drho = np.gradient(rho, axis=0)
    dz = -np.gradient(depth, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        N2 = - (g / rho0) * (drho / dz)
    return N2

# ========== Loop through weekly files ==========
N2ml_list = []
time_list = []

for f in files:
    date_tag = os.path.basename(f).split("_")[-1].replace(".nc", "")
    print(f"\nProcessing: {os.path.basename(f)}")

    # ds = xr.open_dataset(f, chunks={"k": 10, "j": 100, "i": 100})
    ds = xr.open_dataset(f)

    if "rho_7d" not in ds or "Hml_7d" not in ds:
        print(f"Missing variables in {f}, skipping.")
        ds.close()
        continue

    rho = ds["rho_7d"]
    Hml = ds["Hml_7d"]

    # Assign depth as coordinate
    rho = rho.assign_coords(k=depth_1d)

    # Rechunk so that 'k' is not split (required for vertical gradient)
    rho = rho.chunk({'k': -1, 'j': -1, 'i': -1})

    # Compute N² using xarray and apply_ufunc
    depth_broadcasted = xr.DataArray(
        np.broadcast_to(depth_1d[:, None, None], rho.shape),
        dims=rho.dims,
        coords=rho.coords
    )

    depth_broadcasted = depth_broadcasted.chunk({'k': -1, 'j': -1, 'i': -1})

    N2 = xr.apply_ufunc(
        compute_N2,
        rho,
        depth_broadcasted,
        input_core_dims=[["k", "j", "i"], ["k", "j", "i"]],
        output_core_dims=[["k", "j", "i"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    N2 = N2.assign_coords(k=depth_1d)

    # Compute depth bounds (50%-90%) of mixed layer
    Hml_50 = Hml * 0.5
    Hml_90 = Hml * 0.9

    # Interpolate to nearest depth levels
    k_50 = np.abs(depth_1d[:, None, None] - Hml_50.values[None, :, :]).argmin(axis=0)
    k_90 = np.abs(depth_1d[:, None, None] - Hml_90.values[None, :, :]).argmin(axis=0)

    j_idx, i_idx = np.meshgrid(np.arange(k_50.shape[0]), np.arange(k_50.shape[1]), indexing='ij')

    # Mask N² between k_90 and k_50
    N2_selected = []
    for j in range(N2.j.size):
        for i in range(N2.i.size):
            k90 = k_90[j, i]
            k50 = k_50[j, i]
            if k90 <= k50 or k90 >= len(depth_1d):
                continue
            subset = N2.isel(j=j, i=i).isel(k=slice(k50, k90+1)).values
            if np.all(np.isnan(subset)):
                continue
            N2_selected.append(np.nanmean(subset))

    if len(N2_selected) < 100:
        print(f"Too few valid N2 values in {date_tag}, skipping.")
        ds.close()
        continue

    mean_N2 = np.nanmean(N2_selected)
    N2ml_list.append(mean_N2)
    time_list.append(np.datetime64(date_tag))
    ds.close()

# ========== Save to NetCDF ==========
time_da = xr.DataArray(time_list, dims="time", name="time")
N2ml_da = xr.DataArray(N2ml_list, dims="time", name="N2ml_mean", coords={"time": time_da})
out_ds = xr.Dataset({"N2ml_mean": N2ml_da})
out_ds.to_netcdf(output_file)
print(f"\n Saved mixed-layer averaged N² to: {output_file}")



# ========== Imports ==========
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from set_constant import domain_name

# ========== Paths ==========
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/"
os.makedirs(figdir, exist_ok=True)

input_file = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/N2ml_weekly.nc"

# ========== Load Data ==========
ds = xr.open_dataset(input_file)
time = ds["time"].values
N2ml = ds["N2ml_mean"].values
ds.close()

# ========== Plot ==========
plt.figure(figsize=(10, 4))
plt.plot(time, N2ml, marker='o', linestyle='-', color='darkgreen')
plt.ylabel("N² (1/s²)")
plt.yscale('log')
plt.ylim(1e-8, 1e-3)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Mean Stratification (N²) in 50%-90% of the Mixed Layer")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
save_path = os.path.join(figdir, "N2ml_mean_timeseries.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f" Saved plot: {save_path}")

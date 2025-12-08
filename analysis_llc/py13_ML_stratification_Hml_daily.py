### Compute and plot the mean stratification in the ML

# ========== Imports ==========
import os
import numpy as np
import xarray as xr
from glob import glob
from dask.distributed import Client, LocalCluster

from set_constant import domain_name

# ========== Constants ==========
g = 9.81
rho0 = 1027.5
delta_rho = 0.03

# ========== Setup Dask ==========
cluster = LocalCluster(n_workers=20, threads_per_worker=1, memory_limit="18GB")
# cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
output_file = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/", "N2ml_daily.nc")
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
files = sorted(glob(os.path.join(input_dir, "rho_Hml_TS_daily_*.nc")))

# ========== Load grid ==========
print("Loading vertical grid from Zarr...")
ds_grid = xr.open_zarr(zarr_path, consolidated=False)
depth_1d = ds_grid.Z.values
ds_grid.close()


# ========== Compute N² function ==========
def compute_N2_xr(rho, depth):
    drho = - rho.differentiate("k")
    dz = - depth.differentiate("k")
    N2 = - (g / rho0) * (drho / dz)
    return N2

# ========== Loop through weekly files ==========
N2ml_list = []
time_list = []

for f in files:
    date_tag = os.path.basename(f).split("_")[-1].replace(".nc", "")
    print(f"\nProcessing: {os.path.basename(f)}")

    ds = xr.open_dataset(f)

    fname_Hml = f"{input_dir}/Hml_daily_surface_reference_{date_tag}.nc"
    ds_Hml = xr.open_dataset(fname_Hml)

    if "rho_daily" not in ds or "Hml_daily" not in ds_Hml:
        print(f"Missing variables in {f}, skipping.")
        ds.close()
        continue

    rho = ds["rho_daily"]

    Hml = ds_Hml["Hml_daily"]

    # Rechunk so that 'k' is not split (required for vertical gradient)
    rho = rho.chunk({'k': -1, 'j': -1, 'i': -1})

    # Compute N² using xarray and apply_ufunc
    depth_broadcasted = xr.DataArray(
        np.broadcast_to(depth_1d[:, None, None], rho.shape),
        dims=rho.dims,
        coords=rho.coords
    )

    depth_broadcasted = depth_broadcasted.chunk({'k': -1, 'j': -1, 'i': -1})

    N2 = compute_N2_xr(rho, depth_broadcasted)

    # Compute depth bounds (50%-90%) of mixed layer
    # Hml_50 = Hml * 0.5
    # Hml_90 = Hml * 0.9
    Hml_50 = Hml * 0
    Hml_90 = Hml * 1

    # Interpolate to nearest depth levels
    k_50 = np.abs(depth_1d[:, None, None] - Hml_50.values[None, :, :]).argmin(axis=0)
    k_90 = np.abs(depth_1d[:, None, None] - Hml_90.values[None, :, :]).argmin(axis=0)

    # Convert to xarray DataArrays
    k_50_da = xr.DataArray(k_50, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})
    k_90_da = xr.DataArray(k_90, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})

    # Create k indices aligned with N2
    k_indices, _, _ = xr.broadcast(N2["k"], N2["j"], N2["i"])
    k_indices = k_indices.astype(int)

    # Mask for k within 50%-90% of Hml
    in_range_mask = (k_indices >= k_50_da) & (k_indices <= k_90_da)

    # Apply mask
    N2_masked = N2.where(in_range_mask)

    drF_3d, _ = xr.broadcast(ds_grid.drF, rho)
    drF_mask  = drF_3d.where(in_range_mask)

    # Mean over vertical dimension (k)
    # N2ml_mean = N2_masked.mean(dim="k", skipna=True)

    # vertical mean: sum(N² * dz) / sum(dz)
    N2ml_mean = (N2_masked * drF_mask).sum(dim="k", skipna=True) / drF_mask.sum(dim="k", skipna=True)

    # Spatial mean
    mean_N2 = N2ml_mean.isel(i=slice(2,-2),j=slice(2,-2)).mean(dim=["j", "i"], skipna=True).compute()

    N2ml_list.append(mean_N2)
    date_iso = f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:]}"
    time_list.append(np.datetime64(date_iso))
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

# Global font size setting for figures
plt.rcParams.update({'font.size': 15})

from set_constant import domain_name

# ========== Paths ==========
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Hml_tendency"
os.makedirs(figdir, exist_ok=True)

input_file = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/N2ml_daily.nc"

# ========== Load Data ==========
ds = xr.open_dataset(input_file)
time = ds["time"]
N2ml = ds["N2ml_mean"]
ds.close()

# N2ml = N2ml.where(N2ml >= 0, np.nan)
N2ml_rolling = N2ml.rolling(time=7, center=True).mean()
Hml_reconstructed = g*delta_rho/rho0/N2ml
Hml_reconstructed_rolling = g*delta_rho/rho0/N2ml_rolling




# ========== Plot ==========
plt.figure(figsize=(10, 4))
plt.plot(time, N2ml, marker='o', linestyle='-', color='darkgreen')
plt.ylabel("N² (1/s²)")
plt.yscale('log')
# plt.ylim(1e-40, 1e-3)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Mean Stratification (N²) of the Mixed Layer")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
save_path = os.path.join(figdir, "N2ml_timeseries_daily.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f" Saved plot: {save_path}")



fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily.nc"
Hml_mean = abs(xr.open_dataset(fname).Hml_mean)


# ========== Plot ==========
plt.figure(figsize=(10, 6))
plt.plot(time, Hml_reconstructed, label= r'$g\,\Delta\rho/\rho_0/\bar{N}^2$', linestyle='--', color='darkgreen')
plt.plot(time, Hml_reconstructed_rolling, label= r'$g\,\Delta\rho/\rho_0/\bar{N}^2$ with 7-day rolling mean $\bar{N}^2$', linestyle='-', color='green')
plt.plot(Hml_mean.time, Hml_mean, label='mean surface referenced Hml', linestyle='-', color='blue')
plt.ylabel("h (m)")
plt.ylim(0, 1200)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Mixed layer depth")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()
# plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
save_path = os.path.join(figdir, "Hml_reconstructed_using_N2ml.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f" Saved plot: {save_path}")


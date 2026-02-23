# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ========== Time settings ==========
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB",
    processes=True
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
base_out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale"
os.makedirs(base_out_dir, exist_ok=True)

out_nc_path_submeso = os.path.join(base_out_dir, "SSH_Gaussian_submeso_17kmCutoff.nc")
out_nc_path_meso = os.path.join(base_out_dir, "SSH_Gaussian_meso_17kmCutoff.nc")
out_nc_path_meso_coarse = os.path.join(base_out_dir, "SSH_Gaussian_meso_17kmCutoff_1_12deg.nc")

plot_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_submesoscale"
os.makedirs(plot_dir, exist_ok=True)

# =====================
# Reference info
# =====================
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds1 = xr.open_zarr(zarr_path, consolidated=False)
time_ref = ds1.time.isel(time=0).values
total_days = (end_hours - start_hours) // 24
print(f"Total days to process: {total_days}")

# =====================
# Load all daily SSH files (lazy)
# =====================
print("🔄 Loading all daily Eta files...")
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds = xr.open_mfdataset(eta_path, combine='by_coords', parallel=True)
ssh = ds.Eta  # shorthand

# =====================
# Grid spacing (km)
# =====================
dxC_mean = ds1.dxC.isel(face=face, i_g=i, j=j).values.mean() / 1000
dyC_mean = ds1.dyC.isel(face=face, i=i, j_g=j).values.mean() / 1000
dx_km = np.sqrt(0.5 * (dxC_mean**2 + dyC_mean**2))
nyquist_wavelength = 2 * dx_km
print(f"Grid spacing = {dx_km:.2f} km, Nyquist λ = {nyquist_wavelength:.2f} km")

# =====================
# Function: Gaussian filter + coarse-grain mesoscale field
# =====================
@delayed
def process_one_timestep(t_index, date_str, eta_day, dx_km):
    """
    Split SSH into submesoscale (<17 km) and mesoscale (>17 km) components
    using Gaussian filtering, and coarse-grain the mesoscale field to 1/12°.
    """
    try:
        # Remove domain mean
        eta_mean_removed = eta_day - eta_day.mean(dim=["i", "j"])

        # ---- Gaussian filter ----
        sigma_km = 17 / np.sqrt(8 * np.log(2))
        sigma_pts = sigma_km / dx_km
        print(f"[{date_str}] Gaussian sigma = {sigma_pts:.2f} grid pts")

        eta_large = xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=sigma_pts, mode='reflect'),
            eta_mean_removed,
            dask="allowed",
            output_dtypes=[eta_mean_removed.dtype],
        )

        # ---- Submesoscale and mesoscale fields ----
        eta_submeso = eta_mean_removed - eta_large
        eta_submeso.name = "SSH_submesoscale"
        eta_submeso.attrs["description"] = "SSH with > 17 km scales removed via Gaussian filter"
        eta_submeso.attrs["filter_cutoff_km"] = 17

        eta_meso = eta_large.rename("SSH_mesoscale")
        eta_meso.attrs["description"] = "SSH with < 17 km scales removed via Gaussian filter"
        eta_meso.attrs["filter_cutoff_km"] = 17

        # ---- Coarse-grain mesoscale field only ----
        coarse_factor = 4  # 1/48° → 1/12°
        eta_meso_coarse = (
            eta_meso.coarsen(i=coarse_factor, j=coarse_factor, boundary="trim")
            .mean()
            .rename("SSH_mesoscale_coarse")
        )
        eta_meso_coarse.attrs["description"] = "Gaussian-filtered mesoscale SSH coarse-grained to 1/12°"

        return date_str, eta_submeso, eta_meso, eta_meso_coarse

    except Exception as e:
        print(f"⚠️ Error processing {date_str}: {e}")
        nan_field = xr.full_like(eta_day, np.nan)
        nan_coarse = xr.full_like(eta_day.coarsen(i=4, j=4, boundary='trim').mean(), np.nan)
        return date_str, nan_field, nan_field, nan_coarse


# =====================
# Schedule tasks
# =====================
tasks = []
for t in range(ds.dims["time"]):
    date_str = np.datetime_as_string(ds.time.isel(time=t).values, unit="D")
    eta_day = ssh.isel(time=t)
    tasks.append(process_one_timestep(t, date_str, eta_day, dx_km))

print(f"🧮 Scheduled {len(tasks)} Dask tasks...")

# =====================
# Compute in parallel
# =====================
results = compute(*tasks)
dates, ssh_submeso_list, ssh_meso_list, ssh_meso_coarse_list = zip(*results)

# =====================
# Combine results and save
# =====================
time_coords = np.array(dates, dtype="datetime64[D]")

SSH_submesoscale = xr.concat(ssh_submeso_list, dim="time").assign_coords(time=("time", time_coords))
SSH_mesoscale = xr.concat(ssh_meso_list, dim="time").assign_coords(time=("time", time_coords))
SSH_mesoscale_coarse = xr.concat(ssh_meso_coarse_list, dim="time").assign_coords(time=("time", time_coords))

ds_out_submeso = xr.Dataset({"SSH_submesoscale": SSH_submesoscale})
ds_out_meso = xr.Dataset({"SSH_mesoscale": SSH_mesoscale})
ds_out_meso_coarse = xr.Dataset({"SSH_mesoscale_coarse": SSH_mesoscale_coarse})

ds_out_submeso.to_netcdf(out_nc_path_submeso)
ds_out_meso.to_netcdf(out_nc_path_meso)
ds_out_meso_coarse.to_netcdf(out_nc_path_meso_coarse)

print(f"\n✅ Saved submesoscale SSH: {out_nc_path_submeso}")
print(f"✅ Saved mesoscale SSH: {out_nc_path_meso}")
print(f"✅ Saved mesoscale coarse-grained SSH: {out_nc_path_meso_coarse}")

# =====================
# Cleanup
# =====================
ds.close()
ds1.close()
client.close()
cluster.close()
print("🏁 Done!")

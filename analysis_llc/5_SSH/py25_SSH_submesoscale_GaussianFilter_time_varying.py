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
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# =====================
# Time settings
# =====================
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB", processes=True)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
base_out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale"
os.makedirs(base_out_dir, exist_ok=True)

out_nc_path_submeso = os.path.join(base_out_dir, "SSH_Gaussian_submeso_LambdaMLI_GSW0.8.nc")
out_nc_path_meso = os.path.join(base_out_dir, "SSH_Gaussian_meso_LambdaMLI_GSW0.8.nc")
out_nc_path_meso_coarse = os.path.join(base_out_dir, "SSH_Gaussian_meso_LambdaMLI_1_12deg_GSW0.8.nc")

# =====================
# Load Lambda_MLI_mean (km)
# =====================
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily_surface_reference_GSW.nc"
Lambda_MLI_mean = xr.open_dataset(fname).Lambda_MLI_mean / 1000.0

Lambda_MLI_mean = 0.8*Lambda_MLI_mean ### Set the filter scale to be 80% of the most unstable wavelength

# =====================
# Reference grid info
# =====================
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds1 = xr.open_zarr(zarr_path, consolidated=False)

dxC_mean = ds1.dxC.isel(face=face, i_g=i, j=j).values.mean() / 1000
dyC_mean = ds1.dyC.isel(face=face, i=i, j_g=j).values.mean() / 1000
dx_km = np.sqrt(0.5 * (dxC_mean**2 + dyC_mean**2))
print(f"Grid spacing ≈ {dx_km:.2f} km")

# =====================
# Load SSH (lazy)
# =====================
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds = xr.open_mfdataset(eta_path, combine="by_coords", parallel=True)
ssh = ds.Eta

ssh_time_daily = ssh.time.dt.floor("D")

# Align Lambda_MLI_mean to SSH time
# Lambda_MLI_mean = Lambda_MLI_mean.sel(time=ssh.time)
Lambda_MLI_mean = Lambda_MLI_mean.sel(time=ssh_time_daily)


# =====================
# Function: time-varying Gaussian filter
# =====================
@delayed
def process_one_timestep(t_index, date_str, eta_day, lambda_km, dx_km):
    """
    Gaussian filter SSH using a time-varying cutoff scale Lambda_MLI_mean(t)
    """
    try:
        # Remove spatial mean
        eta_mean_removed = eta_day - eta_day.mean(dim=["i", "j"])

        # ---- Time-varying Gaussian width ----
        sigma_km = lambda_km / np.sqrt(8.0 * np.log(2.0))
        sigma_pts = sigma_km / dx_km
        print(f"[{date_str}] Lambda_MLI = {lambda_km:.2f} km " f"(σ = {sigma_pts:.2f} grid pts)")

        # Gaussian filter
        eta_large = xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=sigma_pts, mode="reflect"),
            eta_mean_removed,
            dask="allowed",
            output_dtypes=[eta_mean_removed.dtype],
        )

        # Submesoscale & mesoscale
        eta_submeso = (eta_mean_removed - eta_large).rename("SSH_submesoscale")
        eta_submeso.attrs.update(
            description="SSH with scales larger than Lambda_MLI_mean removed",
            Lambda_MLI_km=lambda_km,
        )

        eta_meso = eta_large.rename("SSH_mesoscale")
        eta_meso.attrs.update(
            description="SSH with scales smaller than Lambda_MLI_mean removed",
            Lambda_MLI_km=lambda_km,
        )

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
    lambda_km = float(Lambda_MLI_mean.isel(time=t).values)
    tasks.append(process_one_timestep(t, date_str, eta_day, lambda_km, dx_km))

print(f"🧮 Scheduled {len(tasks)} tasks")

# =====================
# Compute
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

# =====================
# Save
# =====================
xr.Dataset({"SSH_submesoscale": SSH_submesoscale}).to_netcdf(out_nc_path_submeso)
xr.Dataset({"SSH_mesoscale": SSH_mesoscale}).to_netcdf(out_nc_path_meso)
xr.Dataset({"SSH_mesoscale_coarse": SSH_mesoscale_coarse}).to_netcdf(out_nc_path_meso_coarse)

print("✅ Files written:")
print(out_nc_path_submeso)
print(out_nc_path_meso)
print(out_nc_path_meso_coarse)

# =====================
# Cleanup
# =====================
ds.close()
ds1.close()
client.close()
cluster.close()
print("🏁 Done")

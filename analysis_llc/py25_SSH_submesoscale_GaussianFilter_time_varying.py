# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

from set_constant import domain_name, face, i, j

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

out_nc_path_submeso = os.path.join(base_out_dir, "SSH_Gaussian_submeso_LambdaMLI.nc")
out_nc_path_meso = os.path.join(base_out_dir, "SSH_Gaussian_meso_LambdaMLI.nc")
out_nc_path_meso_coarse = os.path.join(base_out_dir, "SSH_Gaussian_meso_LambdaMLI_1_12deg.nc")

# =====================
# Load Lambda_MLI_mean (km)
# =====================
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_7d_rolling.nc"
Lambda_MLI_mean = xr.open_dataset(fname).Lambda_MLI_mean / 1000.0
Lambda_MLI_mean = Lambda_MLI_mean.rename("Lambda_MLI_mean_km")

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

# Align Lambda_MLI_mean to SSH time
Lambda_MLI_mean = Lambda_MLI_mean.sel(time=ssh.time)

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
        eta_prime = eta_day - eta_day.mean(dim=["i", "j"])

        # ---- Time-varying Gaussian width ----
        sigma_km = lambda_km / np.sqrt(8.0 * np.log(2.0))
        sigma_pts = sigma_km / dx_km

        print(
            f"[{date_str}] Lambda_MLI = {lambda_km:.2f} km "
            f"(σ = {sigma_pts:.2f} grid pts)"
        )

        # Gaussian filter
        eta_large = xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=sigma_pts, mode="reflect"),
            eta_prime,
            dask="allowed",
            output_dtypes=[eta_prime.dtype],
        )

        # Submesoscale & mesoscale
        eta_sub = (eta_prime - eta_large).rename("SSH_submesoscale")
        eta_sub.attrs.update(
            description="SSH with scales larger than Lambda_MLI_mean removed",
            Lambda_MLI_km=lambda_km,
        )

        eta_meso = eta_large.rename("SSH_mesoscale")
        eta_meso.attrs.update(
            description="SSH with scales smaller than Lambda_MLI_mean removed",
            Lambda_MLI_km=lambda_km,
        )

        # ---- Coarse-grain mesoscale ----
        eta_meso_coarse = (
            eta_meso.coarsen(i=4, j=4, boundary="trim")
            .mean()
            .rename("SSH_mesoscale_coarse")
        )
        eta_meso_coarse.attrs["description"] = (
            "Gaussian-filtered mesoscale SSH coarse-grained to 1/12°"
        )

        return date_str, eta_sub, eta_meso, eta_meso_coarse

    except Exception as e:
        print(f"⚠️ Error on {date_str}: {e}")
        nan = xr.full_like(eta_day, np.nan)
        nan_c = xr.full_like(
            eta_day.coarsen(i=4, j=4, boundary="trim").mean(), np.nan
        )
        return date_str, nan, nan, nan_c


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
dates, sub_list, meso_list, meso_c_list = zip(*results)

time_coords = np.array(dates, dtype="datetime64[D]")

SSH_sub = xr.concat(sub_list, dim="time").assign_coords(time=time_coords)
SSH_meso = xr.concat(meso_list, dim="time").assign_coords(time=time_coords)
SSH_meso_c = xr.concat(meso_c_list, dim="time").assign_coords(time=time_coords)

# =====================
# Save
# =====================
xr.Dataset({"SSH_submesoscale": SSH_sub}).to_netcdf(out_nc_path_submeso)
xr.Dataset({"SSH_mesoscale": SSH_meso}).to_netcdf(out_nc_path_meso)
xr.Dataset({"SSH_mesoscale_coarse": SSH_meso_c}).to_netcdf(out_nc_path_meso_coarse)

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

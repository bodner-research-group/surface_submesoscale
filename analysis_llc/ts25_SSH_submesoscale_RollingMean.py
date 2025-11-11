# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import detrend
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

# from set_constant import domain_name, face, i, j

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ========== Time settings ==========
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24


# plt.rcParams.update({'font.size': 16})  # Global font size setting for figures

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(
    n_workers=32,
    threads_per_worker=1,
    memory_limit="11GB",
    processes=True
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
out_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale_20kmCutoff.nc"
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
# Function to compute submesoscale SSH for one time step
# =====================
@delayed
def process_one_timestep(t_index, date_str, eta_day, dx_km):
    """
    Remove >20 km large-scale background using rolling mean.
    Returns (date, SSH_submesoscale)
    """
    try:
        # Remove domain mean
        eta_mean_removed = eta_day - eta_day.mean(dim=["i", "j"])

        # Compute rolling window size in grid points for 20 km
        window_size = int(np.ceil(20.0 / dx_km))
        if window_size % 2 == 0:
            window_size += 1  # ensure odd window size for centered rolling

        # Remove large-scale (>20 km) background
        eta_large = eta_mean_removed.rolling(i=window_size, j=window_size, center=True).mean()
        eta_submeso = eta_mean_removed - eta_large

        # Return as DataArray with proper metadata
        eta_submeso.name = "SSH_submesoscale"
        eta_submeso.attrs["description"] = "SSH with >20 km scales removed"
        eta_submeso.attrs["filter_cutoff_km"] = 20.0

        return date_str, eta_submeso
    except Exception as e:
        print(f"⚠️ Error processing {date_str}: {e}")
        return date_str, xr.full_like(eta_day, np.nan).rename("SSH_submesoscale")

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
dates, ssh_submeso_list = zip(*results)

# =====================
# Combine results and save
# =====================
SSH_submesoscale = xr.concat(ssh_submeso_list, dim="time")
SSH_submesoscale = SSH_submesoscale.assign_coords(time=("time", np.array(dates, dtype="datetime64[D]")))

ds_out = xr.Dataset({"SSH_submesoscale": SSH_submesoscale})
ds_out.to_netcdf(out_nc_path)

print(f"\n✅ Saved submesoscale SSH dataset: {out_nc_path}")

# =====================
# Cleanup
# =====================
ds.close()
ds1.close()
client.close()
cluster.close()
print("🏁 Done!")

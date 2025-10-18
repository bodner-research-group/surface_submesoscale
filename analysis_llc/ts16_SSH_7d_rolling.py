# ===== Imports =====
import os
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, start_hours, end_hours, step_hours

# =====================
# Dask Setup
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
rolling_out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_7d_rolling"
os.makedirs(rolling_out_dir, exist_ok=True)

# =====================
# Gather all weekly files
# =====================
# You assume each weekly file has 7 time steps (1 per day)
all_files = sorted([f for f in os.listdir(eta_input_dir) if f.startswith("eta_24h_") and f.endswith(".nc")])
if not all_files:
    raise FileNotFoundError("No weekly Eta files found.")

# =====================
# Open and concatenate all Eta data
# =====================
print("📂 Opening all Eta weekly files...")
datasets = []
for fname in all_files:
    fpath = os.path.join(eta_input_dir, fname)
    ds = xr.open_dataset(fpath, chunks={'time': 7, 'i': -1, 'j': -1})
    datasets.append(ds)

# Concatenate along time
ds_all = xr.concat(datasets, dim='time')
ds_all = ds_all.sortby('time')  # ensure time is ordered
print(f"📈 Total time steps: {ds_all.time.size}")

# =====================
# Compute 7-day rolling mean
# =====================
print("🔄 Computing 7-day rolling mean...")
eta_rolling = ds_all['Eta'].rolling(time=7, center=True).mean().dropna(dim='time')
eta_rolling = eta_rolling.rename("eta_7d")

# =====================
# Save rolling mean files per day
# =====================
for t in eta_rolling.time.values:
    date_str = str(np.datetime_as_string(t, unit='D'))
    out_path = os.path.join(rolling_out_dir, f"eta_7d_rolling_{date_str}.nc")

    if os.path.exists(out_path):
        print(f"⏩ Skipping existing file: {out_path}")
        continue

    print(f"💾 Saving: {out_path}")
    eta_day = eta_rolling.sel(time=t)
    eta_day = eta_day.expand_dims("time")
    eta_day.to_netcdf(out_path)

# =====================
# Done
# =====================
print("\n✅ 7-day rolling mean Eta files saved.")
client.close()
cluster.close()

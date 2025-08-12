##### Compute and plot weekly mean mixed layer depth

# ========== Imports ==========
import xarray as xr
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# ========= Paths ==========
hml_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_weekly"
output_file = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/", "Hml_weekly_mean.nc")
hml_files = sorted(glob(os.path.join(hml_dir, "rho_Hml_7d_*.nc")))

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/"

weekly_hml_means = []
time_list = []

# ========= Load and compute mean Hml ==========
for f in hml_files:
    ds = xr.open_dataset(f)
    
    if "Hml_7d" in ds:
        hml = ds["Hml_7d"].load()
        hml_mean = hml.mean().item()
        weekly_hml_means.append(hml_mean)
        
        date_tag = os.path.basename(f).split("_")[-1].replace(".nc", "")
        time_list.append(pd.to_datetime(date_tag))
        
        ds.close()
    else:
        print(f"Warning: 'Hml_7d' not found in {f}")

# ========= Create xarray Dataset ==========
time = xr.DataArray(time_list, dims="time", name="time")
Hml_mean = xr.DataArray(weekly_hml_means, dims="time", name="Hml_mean", coords={"time": time})

ds_out = xr.Dataset({"Hml_mean": Hml_mean})

# ========= Save to NetCDF ==========
ds_out.to_netcdf(output_file)
print(f"Saved weekly Hml means to: {output_file}")








# Global font size setting for figures
plt.rcParams.update({'font.size': 16})


import matplotlib.dates as mdates

def plot_timeseries(times, values, ylabel, save_name, yscale='linear'):
    plt.figure(figsize=(10, 4))
    plt.plot(times, values, marker='o', linestyle='-', label=ylabel)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.grid(True, linestyle='--')
    plt.title(f"{ylabel} over Time")
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, save_name), dpi=150)
    plt.close()
    print(f"Saved time series plot: {save_name}")

# plot
plot_timeseries(Hml_mean.time.values, Hml_mean.values, "Mean ML Depth (m²/s³)", "Hml_mean_timeseries.png")

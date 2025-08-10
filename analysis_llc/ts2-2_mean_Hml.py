##### Time series of the following variables:
#####
##### Qnet (net surface heat flux into the ocean),
##### Hml (mixed-layer depth), 
##### TuH (horizontal Turner angle), 
##### TuV (vertical Turner angle),
##### wb_cros (variance-perserving cross-spectrum of vertical velocity and buoyancy)
##### wbmin (the minimum of wb_cros)
##### Lmax (the horizontal length scale corresponds to wbmin), 
##### Dmax (the depth corresponds to wbmin), 
##### gradSSH (absolute gradient of sea surface height anomaly), etc.
#####
##### Step 1: compute 24-hour averages of temperature, salinity, and vertical velocity, save as .nc files
##### Step 2: compute 7-day averages of potential density, alpha, beta, Hml, save as .nc files
##### Step 3: compute wb_cros using the 24-hour averages, and then compute the 7-day averaged wb_cros 
##### Step 4: plot wb_cros of each week, compute wbmin, Lmax, Dmax
##### Step 5: compute 7-day movmean of Qnet
##### Step 6: compute TuH and TuV

# ========== Imports ==========
import xarray as xr
import os
from glob import glob
import numpy as np
import pandas as pd

# ========= Paths ==========
hml_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/rho_weekly"
output_file = os.path.join("/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/", "Hml_weekly_mean.nc")
hml_files = sorted(glob(os.path.join(hml_dir, "rho_Hml_7d_*.nc")))

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


import matplotlib.dates as mdates

def plot_timeseries(var, values, ylabel, save_name, yscale='linear'):
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
    plt.savefig(os.path.join(timeseries_figdir, save_name), dpi=150)
    plt.close()
    print(f"Saved time series plot: {save_name}")

# plot
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/wb_spectra_weekly"
timeseries_figdir = os.path.join(figdir, "summary_timeseries")
plot_timeseries("Hml_mean", Hml_mean, "Mean ML Depth (m²/s³)", "Hml_mean_timeseries.png", yscale='log')

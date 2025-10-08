import os
import xarray as xr
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os

from set_constant import domain_name, face, i, j
plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

# --- Set paths ---
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI"
out_timeseries_path = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/", "Lambda_MLI_timeseries.nc")

# --- Get list of output files ---
nc_files = sorted(glob(os.path.join(output_dir, "Lambda_MLI_*.nc")))

# --- Initialize lists to store results ---
dates = []
Lambda_MLI_mean_list = []
M2_mean_list = []
N2ml_mean_list = []
Hml_mean_list = []

# --- Loop over all output files ---
for fpath in nc_files:
    print(f"Processing {os.path.basename(fpath)}")
    ds = xr.open_dataset(fpath)

    # Extract date tag from filename
    date_tag = os.path.basename(fpath).split("_")[-1].replace(".nc", "")
    try:
        time_val = np.datetime64(date_tag)
    except ValueError:
        time_val = date_tag  # fallback to string if parsing fails

    # Subset inner domain: exclude 2 points on all sides
    inner_slice = dict(i=slice(2, -2), j=slice(2, -2))

    Lambda_MLI_inner = ds["Lambda_MLI"].isel(**inner_slice)
    Mml4_mean_inner = ds["Mml4_mean"].isel(**inner_slice)
    N2ml_mean_inner = ds["N2ml_mean"].isel(**inner_slice)
    Hml_inner = ds["Hml"].isel(**inner_slice)

    # Compute M2 = sqrt(Mml4_mean)
    M2_inner = np.sqrt(Mml4_mean_inner)

    # Compute domain means
    Lambda_MLI_mean = Lambda_MLI_inner.mean(dim=("j", "i"), skipna=True)
    M2_mean = M2_inner.mean(dim=("j", "i"), skipna=True)
    N2ml_mean = N2ml_mean_inner.mean(dim=("j", "i"), skipna=True)
    Hml_mean = Hml_inner.mean(dim=("j", "i"), skipna=True)

    # Append results
    dates.append(time_val)
    Lambda_MLI_mean_list.append(Lambda_MLI_mean.values)
    M2_mean_list.append(M2_mean.values)
    N2ml_mean_list.append(N2ml_mean.values)
    Hml_mean_list.append(Hml_mean.values)

# --- Convert to xarray Dataset ---
times = np.array(dates)
ds_out = xr.Dataset(
    {
        "Lambda_MLI_mean": (("time",), Lambda_MLI_mean_list),
        "M2_mean": (("time",), M2_mean_list),
        "N2ml_mean": (("time",), N2ml_mean_list),
        "Hml_mean": (("time",), Hml_mean_list),
    },
    coords={"time": times},
)

# --- Save to NetCDF ---
encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
ds_out.to_netcdf(out_timeseries_path, encoding=encoding)

print(f"✅ Saved domain-averaged timeseries (excluding boundaries) to {out_timeseries_path}")




# --- Path to timeseries file ---
fig_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Lambda_MLI"
out_timeseries_path = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/", "Lambda_MLI_timeseries.nc")

# --- Load dataset ---
ds = xr.open_dataset(out_timeseries_path)

# --- Convert Lambda_MLI to km for plotting ---
Lambda_km = ds["Lambda_MLI_mean"] / 1000

# --- Plot all time series ---
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)

axs[0].plot(ds.time, Lambda_km, color='blue')
axs[0].set_ylabel("Lambda_MLI (km)")
axs[0].set_title("Domain-Averaged MLI Diagnostics")

axs[1].plot(ds.time, ds["M2_mean"], color='green')
axs[1].set_ylabel("M² (s⁻²)")

axs[2].plot(ds.time, ds["N2ml_mean"], color='purple')
axs[2].set_ylabel("N² (s⁻²)")

axs[3].plot(ds.time, -ds["Hml_mean"], color='orange')
axs[3].set_ylabel("Hml (m)")
axs[3].set_xlabel("Time")

# Optional: add gridlines
for ax in axs:
    ax.grid(True)

# --- Save figure ---
plot_path = os.path.join(fig_dir, "MLI_timeseries_plot.png")
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"✅ Time series plot saved to: {plot_path}")

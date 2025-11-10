import os
import xarray as xr
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

plt.rcParams.update({'font.size': 16})  # Global font size setting for figures

# --- Paths ---
# output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI"
# out_timeseries_path = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}", "Lambda_MLI_timeseries_7d_rolling.nc")
# fig_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Lambda_MLI"
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_daily"
out_timeseries_path = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}", "Lambda_MLI_timeseries_daily.nc")
fig_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Lambda_MLI_daily"
os.makedirs(fig_dir, exist_ok=True)

# --- Get list of output files ---
nc_files = sorted(glob(os.path.join(output_dir, "Lambda_MLI_*.nc")))

# --- Initialize lists ---
dates = []
Lambda_MLI_mean_list = []
M2_mean_list = []
M2_mean_new_list = []
N2ml_mean_list = []
Hml_mean_list = []

# --- Loop over files ---
for fpath in nc_files:
    print(f"Processing {os.path.basename(fpath)}")
    ds = xr.open_dataset(fpath, decode_times=False)

    # Extract date from filename
    date_tag = os.path.basename(fpath).split("_")[-1].replace(".nc", "")
    try:
        # Try YYYYMMDD format
        time_val = np.datetime64(f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:8]}")
    except Exception:
        time_val = date_tag  # fallback to string if cannot parse

    # Subset inner domain: exclude 2 points on each side
    inner_slice = dict(i=slice(2, -2), j=slice(2, -2))

    Lambda_MLI_inner = np.abs(ds.Lambda_MLI.isel(**inner_slice))
    Mml4_mean_inner = ds.Mml4_mean.isel(**inner_slice)
    Mml2_mean_inner = ds.Mml2_mean.isel(**inner_slice)
    N2ml_mean_inner = ds.N2ml_mean.isel(**inner_slice)
    Hml_inner = ds.Hml.isel(**inner_slice)

    # Compute M2
    M2_inner = np.sqrt(Mml4_mean_inner)

    # Compute domain mean
    Lambda_MLI_mean = Lambda_MLI_inner.mean(dim=("j", "i"), skipna=True)
    M2_mean = M2_inner.mean(dim=("j", "i"), skipna=True)
    M2_mean_new = Mml2_mean_inner.mean(dim=("j", "i"), skipna=True)
    N2ml_mean = N2ml_mean_inner.mean(dim=("j", "i"), skipna=True)
    Hml_mean = Hml_inner.mean(dim=("j", "i"), skipna=True)

    # Append
    dates.append(time_val)
    Lambda_MLI_mean_list.append(Lambda_MLI_mean.values)
    M2_mean_list.append(M2_mean.values)
    M2_mean_new_list.append(M2_mean_new.values)
    N2ml_mean_list.append(N2ml_mean.values)
    Hml_mean_list.append(Hml_mean.values)

# --- Convert to xarray Dataset ---
times = np.array(dates, dtype='datetime64[D]')
ds_out = xr.Dataset(
    {
        "Lambda_MLI_mean": (("time",), Lambda_MLI_mean_list),
        "M2_mean": (("time",), M2_mean_list),
        "M2_mean_new": (("time",), M2_mean_new_list),
        "N2ml_mean": (("time",), N2ml_mean_list),
        "Hml_mean": (("time",), Hml_mean_list),
    },
    coords={"time": times},
)

# --- Save to NetCDF ---
encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
ds_out.to_netcdf(out_timeseries_path, encoding=encoding)
print(f"✅ Saved domain-averaged timeseries to {out_timeseries_path}")

# --- Plotting ---
ds = xr.open_dataset(out_timeseries_path)  # decode_times now works because times are valid
Lambda_km = ds.Lambda_MLI_mean / 1000

fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)

axs[0].plot(ds.time, Lambda_km, color='blue')
axs[0].set_ylabel("Lambda_MLI (km)")
axs[0].set_title("Domain-Averaged MLI Diagnostics")

axs[1].plot(ds.time, ds["M2_mean"], color='green')
axs[1].plot(ds.time, ds["M2_mean_new"], color='red')
axs[1].legend(["M2_mean", "M2_mean_new"])
axs[1].set_ylabel("M² (s⁻²)")

axs[2].plot(ds.time, ds["N2ml_mean"], color='purple')
axs[2].set_ylabel("N² (s⁻²)")

axs[3].plot(ds.time, -ds["Hml_mean"], color='orange')
axs[3].set_ylabel("Hml (m)")
axs[3].set_xlabel("Time")

for ax in axs:
    ax.grid(True)

plot_path = os.path.join(fig_dir, "MLI_timeseries_plot.png")
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"✅ Time series plot saved to: {plot_path}")
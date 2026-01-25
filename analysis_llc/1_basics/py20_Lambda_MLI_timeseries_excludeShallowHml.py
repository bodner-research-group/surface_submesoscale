import os
import xarray as xr
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

plt.rcParams.update({'font.size': 16})

# --- Paths ---
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_daily_surface_reference"
out_timeseries_path = os.path.join(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}",
    "Lambda_MLI_timeseries_daily_surface_reference.nc"
)
fig_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/Lambda_MLI_daily_surface_reference"
os.makedirs(fig_dir, exist_ok=True)

# --- Get list of output files ---
nc_files = sorted(glob(os.path.join(output_dir, "Lambda_MLI_*.nc")))

# --- Initialize lists ---
dates = []
Lambda_MLI_mean_list = []
Lambda_MLI_mean_km_list = []
M2_mean_list = []
M2_mean_new_list = []
N2ml_mean_list = []
Hml_mean_list = []

# --- Loop over files ---
for fpath in nc_files:
    print(f"Processing {os.path.basename(fpath)}")
    ds = xr.open_dataset(fpath, decode_times=False)

    # --- Extract date ---
    date_tag = os.path.basename(fpath).split("_")[-1].replace(".nc", "")
    time_val = np.datetime64(f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:8]}")

    # --- Inner domain ---
    inner_slice = dict(i=slice(2, -2), j=slice(2, -2))

    Lambda_MLI = np.abs(ds.Lambda_MLI.isel(**inner_slice))
    Mml4 = ds.Mml4_mean.isel(**inner_slice)
    Mml2 = ds.Mml2_mean.isel(**inner_slice)
    N2ml = ds.N2ml_mean.isel(**inner_slice)
    Hml = ds.Hml.isel(**inner_slice)

    # --------------------------------------------------
    # NEW: Mask out shallow mixed layers |Hml| < 10 m
    # --------------------------------------------------
    deep_mask = Hml <= -10.0
    # deep_mask = Hml <= 0

    Lambda_MLI = Lambda_MLI.where(deep_mask)
    Mml4 = Mml4.where(deep_mask)
    Mml2 = Mml2.where(deep_mask)
    N2ml = N2ml.where(deep_mask)
    Hml = Hml.where(deep_mask)

    # --- Diagnostics ---
    M2 = np.sqrt(Mml4)

    # --- Domain means ---
    Lambda_mean = Lambda_MLI.mean(dim=("j", "i"), skipna=True)
    M2_mean = M2.mean(dim=("j", "i"), skipna=True)
    M2_mean_new = Mml2.mean(dim=("j", "i"), skipna=True)
    N2ml_mean = N2ml.mean(dim=("j", "i"), skipna=True)
    Hml_mean = Hml.mean(dim=("j", "i"), skipna=True)

    # --- Append ---
    dates.append(time_val)
    Lambda_MLI_mean_list.append(Lambda_mean.values)
    Lambda_MLI_mean_km_list.append((Lambda_mean / 1000).values)
    M2_mean_list.append(M2_mean.values)
    M2_mean_new_list.append(M2_mean_new.values)
    N2ml_mean_list.append(N2ml_mean.values)
    Hml_mean_list.append(Hml_mean.values)

# --- Convert to Dataset ---
times = np.array(dates, dtype="datetime64[D]")

ds_out = xr.Dataset(
    {
        "Lambda_MLI_mean": (("time",), Lambda_MLI_mean_list),
        "Lambda_MLI_mean_km": (("time",), Lambda_MLI_mean_km_list),
        "M2_mean": (("time",), M2_mean_list),
        "M2_mean_new": (("time",), M2_mean_new_list),
        "N2ml_mean": (("time",), N2ml_mean_list),
        "Hml_mean": (("time",), Hml_mean_list),
    },
    coords={"time": times},
)

# --------------------------------------------------
# NEW: Time-average Lambda from 2012-01-01 to 2012-03-01
# --------------------------------------------------
lambda_winter_mean = (
    ds_out["Lambda_MLI_mean_km"]
    .sel(time=slice("2012-01-01", "2012-03-01"))
    .mean(skipna=True)
)

ds_out.attrs["Lambda_MLI_Jan01_to_Mar01_2012_mean_km"] = float(lambda_winter_mean)

print(
    "✅ Time-mean Lambda_MLI (Jan 01 – Mar 01 2012): "
    f"{lambda_winter_mean.values:.2f} km"
)

# --- Save NetCDF ---
encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
ds_out.to_netcdf(out_timeseries_path, encoding=encoding)
print(f"✅ Saved domain-averaged timeseries to {out_timeseries_path}")

# --- Plotting ---
ds = xr.open_dataset(out_timeseries_path)

fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)

axs[0].plot(ds.time, ds.Lambda_MLI_mean_km, color="blue")
axs[0].set_ylabel("Lambda_MLI (km)")
axs[0].set_title("Domain-Averaged MLI Diagnostics (|Hml| ≥ 10 m)")

axs[1].plot(ds.time, ds.M2_mean, color="green")
axs[1].plot(ds.time, ds.M2_mean_new, color="red")
axs[1].legend(["sqrt(<M⁴>)", "<M²>"])
axs[1].set_ylabel("M² (s⁻²)")

axs[2].plot(ds.time, ds.N2ml_mean, color="purple")
axs[2].set_ylabel("N² (s⁻²)")

axs[3].plot(ds.time, -ds.Hml_mean, color="orange")
axs[3].set_ylabel("Hml (m)")
axs[3].set_xlabel("Time")

for ax in axs:
    ax.grid(True)

plot_path = os.path.join(fig_dir, "MLI_timeseries_plot.png")
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"✅ Time series plot saved to: {plot_path}")



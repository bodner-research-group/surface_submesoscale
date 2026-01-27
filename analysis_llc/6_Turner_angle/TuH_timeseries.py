import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# ===================
# Paths
# ===================
domain_name = "icelandic_basin"

base_dir = (
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/"
    f"analysis_llc/data/{domain_name}/VHF_theory"
)

in_file = os.path.join(base_dir, "TuH_deg_7d_rolling_all.nc")

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/VHF_theory"

fig_file = os.path.join(figdir, "TuH_domain_avg_timeseries.png")
ts_nc_file = os.path.join(base_dir, "TuH_domain_avg_timeseries.nc")

# ===================
# Load dataset
# ===================
ds = xr.open_dataset(in_file)
TuH = ds["TuH_deg"]  # degrees

# ===================
# Exclude 2-point boundary
# ===================
spatial_dims = [d for d in TuH.dims if d != "time"]

TuH_inner = TuH.isel(
    {d: slice(2, -2) for d in spatial_dims}
)

# ===================
# Convert to radians
# ===================
TuH_rad = np.deg2rad(TuH_inner)

# ===================
# 1. < tan(TuH) + 1 >
# ===================
tan_TuH_plus1 = np.tan(TuH_rad) + 1.0
domain_avg_1 = tan_TuH_plus1.mean(dim=spatial_dims)

# ===================
# 2. tan( < TuH > ) + 1
# ===================
domain_avg_TuH = TuH_rad.mean(dim=spatial_dims)
domain_avg_2 = np.tan(domain_avg_TuH) + 1.0

# ===================
# Save time series
# ===================
ds_ts = xr.Dataset(
    {
        "mean_tan_TuH_plus1": domain_avg_1,
        "tan_mean_TuH_plus1": domain_avg_2,
    }
)

ds_ts.to_netcdf(ts_nc_file)


print(f"Saved time series to:")
print(f"  NetCDF: {ts_nc_file}")

# ===================
# Plot & save figure
# ===================
plt.figure(figsize=(10, 5))

plt.plot(
    domain_avg_1["time"],
    domain_avg_1,
    label=r"$\langle \tan(\mathrm{TuH}) + 1 \rangle$",
    linewidth=2,
)

plt.plot(
    domain_avg_2["time"],
    domain_avg_2,
    label=r"$\tan(\langle \mathrm{TuH} \rangle) + 1$",
    linewidth=2,
    linestyle="--",
)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Domain-averaged TuH diagnostics (2-point boundary excluded)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(fig_file, dpi=300)
plt.close()

print(f"Saved figure to: {fig_file}")

ds.close()

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# from set_constant import domain_name
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ==============================================================
# Paths
# ==============================================================
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
# out_dir = os.path.join(base_dir, "steric_height_anomaly_timeseries")
out_dir = os.path.join(base_dir, "steric_height_anomaly_timeseries_surface_reference")

figdir = (
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/"
    f"figs/{domain_name}/steric_height_surface_reference/"
)
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Find daily files
# ==============================================================
file_list = sorted(
    glob(os.path.join(out_dir, "grad_laplace_eta_steric_*.nc"))
)

if len(file_list) == 0:
    raise FileNotFoundError(
        f"No grad_laplace_eta_steric_*.nc found in {out_dir}"
    )

print(f"✅ Found {len(file_list)} daily files")

# ==============================================================
# Containers
# ==============================================================
times = []
eta_grad2_mean_list = []
eta_prime_grad2_mean_list = []

# ==============================================================
# Loop over day files
# ==============================================================
for f in tqdm(file_list, desc="Computing domain-mean |∇η|²"):
    ds = xr.open_dataset(f)

    # ---- time ----
    if "time" in ds.coords:
        time_val = ds.time.values[0]
    else:
        raise ValueError(f"No time coordinate in {f}")

    # ---- gradient magnitudes ----
    eta_grad_mag = ds["eta_grad_mag"]
    eta_prime_grad_mag = ds["eta_prime_grad_mag"]

    # ---- exclude 2 grid points at boundaries ----
    eta_grad_mag = eta_grad_mag.isel(i=slice(2, -2), j=slice(2, -2))
    eta_prime_grad_mag = eta_prime_grad_mag.isel(i=slice(2, -2), j=slice(2, -2))

    # ---- domain means ----
    eta_grad2_mean = (eta_grad_mag ** 2).mean(dim=("i", "j"))
    eta_prime_grad2_mean = (eta_prime_grad_mag ** 2).mean(dim=("i", "j"))

    eta_grad2_mean_list.append(eta_grad2_mean)
    eta_prime_grad2_mean_list.append(eta_prime_grad2_mean)
    times.append(time_val)

    ds.close()

# ==============================================================
# Combine into time series dataset
# ==============================================================
ts_ds = xr.Dataset(
    {
        "eta_grad2_mean": xr.concat(eta_grad2_mean_list, dim="time"),
        "eta_prime_grad2_mean": xr.concat(eta_prime_grad2_mean_list, dim="time"),
    },
    coords={"time": ("time", times)},
)

out_ts_file = os.path.join(out_dir, "grad2_timeseries.nc")
ts_ds.to_netcdf(out_ts_file)

print(f"✅ Saved: {out_ts_file}")

# ==============================================================
# Plot
# ==============================================================
plt.figure(figsize=(8, 4))

ts_ds["eta_grad2_mean"].plot(
    label="⟨|∇η|²⟩", color="tab:blue"
)
ts_ds["eta_prime_grad2_mean"].plot(
    label="⟨|∇η′|²⟩", color="tab:orange"
)

# 7-day rolling mean
ts_ds["eta_grad2_mean"].rolling(time=7, center=True).mean().plot(
    linestyle="--", color="tab:blue", label="⟨|∇η|²⟩ (7d)"
)
ts_ds["eta_prime_grad2_mean"].rolling(time=7, center=True).mean().plot(
    linestyle="--", color="tab:orange", label="⟨|∇η′|²⟩ (7d)"
)

plt.title("Domain-mean ⟨|∇η|²⟩ and ⟨|∇η′|²")
plt.ylabel("m² m⁻²")
plt.xlabel("Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

figfile = os.path.join(figdir, "grad2_timeseries.png")
plt.savefig(figfile, dpi=150)
plt.close()

print(f"✅ Saved figure: {figfile}")

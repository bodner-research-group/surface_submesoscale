import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime

# ========================================
# Settings
# ========================================
from set_constant import domain_name, face, i, j

output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/LHS_integrated_buoyancy_tendency"
fig_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/LHS_integrated_buoyancy_tendency"
os.makedirs(fig_dir, exist_ok=True)


ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lat = ds1['YC'].isel(face=face, i=i, j=j)
lon = ds1['XC'].isel(face=face, i=i, j=j)

# ========================================
# Load all LHS files
# ========================================
lhs_files = sorted(glob(os.path.join(output_dir, "LHS_*.nc")))
date_tags = [os.path.basename(f).split("_")[1].replace(".nc","") for f in lhs_files]

# Only first 30 days
lhs_files_30 = lhs_files[:180]
date_tags_30 = date_tags[:180]

# ========================================
# Pass 1: compute global color limits
# ========================================
vmin = np.inf
vmax = -np.inf

for f in lhs_files_30:
    ds = xr.open_dataset(f)

    for var in ["LHS_true", "LHS_bs"]:
        data = ds[var]
        vmin = min(vmin, float(data.min()))
        vmax = max(vmax, float(data.max()))

    ds.close()

vabs = max(abs(vmin), abs(vmax))
vmin, vmax = -vabs/10, vabs/10

print(f"Global colorbar limits: vmin={vmin:.3e}, vmax={vmax:.3e}")


# ========================================
# Pass 2: plotting
# ========================================
for f, tag in zip(lhs_files_30, date_tags_30):
    ds = xr.open_dataset(f)

    LHS_true = ds["LHS_true"]
    LHS_bs   = ds["LHS_bs"]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # ---- Panel 1: LHS_true ----
    im0 = axs[0].pcolormesh(
        lon,
        lat,
        LHS_true,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="auto"
    )
    axs[0].set_title(f"LHS_true {tag}")
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")

    # ---- Panel 2: LHS_bs ----
    im1 = axs[1].pcolormesh(
        lon,
        lat,
        LHS_bs,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="auto"
    )
    axs[1].set_title(f"LHS_bs {tag}")
    axs[1].set_xlabel("Longitude")

    # ---- Shared colorbar ----
    cbar = fig.colorbar(
        im0,
        ax=axs,
        orientation="vertical",
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label("Integrated buoyancy tendency")

    # ---- Save ----
    fig_file = os.path.join(fig_dir, f"LHS_maps_{tag}.png")
    plt.savefig(fig_file, dpi=150)
    plt.close(fig)

    ds.close()
    print(f"Saved map figure → {fig_file}")


##### Convert images to video
import os
output_movie = f"{fig_dir}/LHS_true_bs.mp4"
os.system(f"ffmpeg -r 15 -pattern_type glob -i '{fig_dir}/LHS_maps_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

print(f"Movie saved → {output_movie}")


# # ========================================
# # Plot 2-panel map figure for first 30 days
# # ========================================
# for f, tag in zip(lhs_files_30, date_tags_30):
#     ds = xr.open_dataset(f)
    
#     LHS_true = ds["LHS_true"]
#     LHS_bs = ds["LHS_bs"]

#     fig, axs = plt.subplots(1,2, figsize=(14,6))
#     im0 = axs[0].pcolormesh(ds["i"], ds["j"], LHS_true, cmap="RdBu_r")
#     axs[0].set_title(f"LHS_true {tag}")
#     plt.colorbar(im0, ax=axs[0])

#     im1 = axs[1].pcolormesh(ds["i"], ds["j"], LHS_bs, cmap="RdBu_r")
#     axs[1].set_title(f"LHS_bs {tag}")
#     plt.colorbar(im1, ax=axs[1])

#     plt.tight_layout()
#     fig_file = os.path.join(fig_dir, f"LHS_maps_{tag}.png")
#     plt.savefig(fig_file, dpi=150)
#     plt.close(fig)
#     print(f"Saved map figure → {fig_file}")

# ========================================
# Compute domain-averaged timeseries
# ========================================
LHS_true_avg = []
LHS_bs_avg = []
dates = []

for f, tag in zip(lhs_files, date_tags):
    ds = xr.open_dataset(f)
    
    # Exclude 2 boundary points
    LHS_true_cut = ds["LHS_true"].isel(i=slice(2,-2), j=slice(2,-2))
    LHS_bs_cut   = ds["LHS_bs"].isel(i=slice(2,-2), j=slice(2,-2))

    # Compute mean, skipping NaNs
    LHS_true_avg.append(float(LHS_true_cut.mean(skipna=True)))
    LHS_bs_avg.append(float(LHS_bs_cut.mean(skipna=True)))
    dates.append(datetime.strptime(tag, "%Y%m%d"))

# Convert to xarray Dataset
ds_avg = xr.Dataset(
    data_vars=dict(
        LHS_true_avg=("time", LHS_true_avg),
        LHS_bs_avg=("time", LHS_bs_avg),
    ),
    coords=dict(time=dates),
    attrs=dict(description="Domain-averaged LHS_true and LHS_bs")
)

# Save timeseries
ts_file = os.path.join(output_dir, "LHS_domain_avg_timeseries.nc")
ds_avg.to_netcdf(ts_file)
print(f"Saved timeseries → {ts_file}")

# ========================================
# Plot timeseries
# ========================================
plt.figure(figsize=(12,5))
plt.plot(dates, LHS_true_avg, label="LHS_true")
plt.plot(dates, LHS_bs_avg, label="LHS_bs")
plt.xlabel("Date")
plt.ylabel("Domain-averaged LHS [m^2/s^3]")
plt.title("Domain-averaged Mixed-layer Integrated Buoyancy Tendency")
plt.legend()
plt.grid(True)
plt.tight_layout()

ts_fig_file = os.path.join(fig_dir, "LHS_domain_avg_timeseries.png")
plt.savefig(ts_fig_file, dpi=150)
plt.close()
print(f"Saved timeseries figure → {ts_fig_file}")

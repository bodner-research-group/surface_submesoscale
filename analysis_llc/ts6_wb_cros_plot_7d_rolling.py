import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import pandas as pd
import matplotlib.dates as mdates

from glob import glob
from set_colormaps import WhiteBlueGreenYellowRed
from set_constant import domain_name

# Set up colormap
cmap = WhiteBlueGreenYellowRed()

# Directories
input_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_cross_spectra_weekly/wb_cross_spec_vp_real_7day_rolling.nc"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_spectra_weekly_7d_rolling"
out_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_max_spec_vp_filtered_7d_rolling_mean.nc"

os.makedirs(figdir, exist_ok=True)
plt.rcParams.update({'font.size': 15})

# Load mixed layer depth
f_Hml = f'/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Hml_weekly_mean.nc'
Hml_mean = xr.open_dataset(f_Hml).Hml_mean  # (time,)

# Load spectrum
ds = xr.open_dataset(input_path)
spec_vp_real = ds.spec_vp_real  # (time, Z, freq_r)
depth = ds.depth
k_r = ds.k_r  # (freq_r,)
times = ds.time.values

# Constants
kr_cutoff = 2 / 500      # cpkm (exclude wavelengths > 500 km)
kr_cutoff_meso = 1 / 30  # cpkm (submeso: < 30 km)

# Prepare storage for results
max_values, max_depths, max_krs, max_Lr = [], [], [], []
mean_spec_in_mld, mean_spec_in_mld_submeso = [], []

# Loop over time
for t_idx, time_val in enumerate(times):
    date_str = np.datetime_as_string(time_val, unit='D')
    print(f"Processing {date_str}...")

    # Extract 2D slice
    spec_2d = spec_vp_real.isel(time=t_idx)
    kr_vals = k_r.values
    depth_vals = depth.values

    # Filter out large-scale (low wavenumber) components
    spec_filtered = spec_2d.where(k_r >= kr_cutoff, drop=True)
    k_r_filtered = k_r.where(k_r >= kr_cutoff, drop=True)

    # Depth mask
    valid_depth_mask = (depth >= Hml_mean.min().values) & (depth <= 0)
    spec_filtered = spec_filtered.where(valid_depth_mask, drop=True)

    # Get HML value this week
    try:
        week_idx = np.where(Hml_mean.time.values == np.datetime64(time_val))[0][0]
        Hml_this_week = Hml_mean.isel(time=week_idx).item()
    except IndexError:
        Hml_this_week = np.nan

    # Mean spectrum in MLD
    if not np.isnan(Hml_this_week):
        mld_mask = depth >= Hml_this_week
        spec_in_mld = spec_filtered.where(mld_mask, drop=True)
        mean_val = spec_in_mld.mean().item()
    else:
        mean_val = np.nan
    mean_spec_in_mld.append(mean_val)

    # Submesoscale portion
    spec_submeso = spec_2d.where(k_r >= kr_cutoff_meso, drop=True)
    spec_submeso = spec_submeso.where(valid_depth_mask, drop=True)
    if not np.isnan(Hml_this_week):
        mld_mask = depth >= Hml_this_week
        spec_in_mld_submeso = spec_submeso.where(mld_mask, drop=True)
        mean_val_submeso = spec_in_mld_submeso.mean().item()
    else:
        mean_val_submeso = np.nan
    mean_spec_in_mld_submeso.append(mean_val_submeso)

    # Find minimum (strongest negative contribution)
    min_val = spec_filtered.min().item()
    idx = np.unravel_index(np.argmin(spec_filtered.values), spec_filtered.shape)
    depth_at_min = spec_filtered.Z[idx[0]].item()
    kr_at_min = k_r_filtered[idx[1]].item()
    Lr_at_min = 1 / kr_at_min  # in km

    # Store
    max_values.append(min_val)
    max_depths.append(depth_at_min)
    max_krs.append(kr_at_min)
    max_Lr.append(Lr_at_min)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = TwoSlopeNorm(vcenter=1e-10, vmin=0, vmax=5e-10)
    pc = ax.pcolormesh(
        k_r_filtered, spec_filtered.Z, spec_filtered,
        shading='auto', cmap=cmap, norm=norm
    )
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_title(f'$w$-$b$ Cross-Spectrum (VP), {date_str}')
    ax.set_xlabel('Wavenumber $k_r$ (cpkm)')
    ax.set_ylabel('Depth (m)')
    fig.colorbar(pc, ax=ax, label=r'Spectral density (m$^2$s$^{-3}$)')
    ax.axvline(kr_cutoff_meso, color='r', linestyle='--', label='30 km cutoff')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figdir}/wb_cross-spectrum_{date_str}.png", dpi=150)
    plt.close()

print("✅ All weekly plots saved.")

# ========== Save results to NetCDF ==========
summary_ds = xr.Dataset(
    {
        "wbmin": (["time"], max_values),
        "depth_at_min": (["time"], max_depths),
        "kr_at_min": (["time"], max_krs),
        "Lr_at_min": (["time"], max_Lr),
        "mean_spec_in_MLD": (["time"], mean_spec_in_mld),
        "mean_spec_in_MLD_submeso": (["time"], mean_spec_in_mld_submeso),
    },
    coords={"time": times}
)
summary_ds.to_netcdf(out_nc_path)
print(f"✅ Saved summary NetCDF to: {out_nc_path}")

# ========== Plot summary timeseries ==========
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# (a) Minimum value (most negative contribution)
axs[0].plot(times, max_values, marker='o', color='tab:blue')
axs[0].set_title("(a) Minimum Spectral Power (wbmin)")
axs[0].set_ylabel("wbmin (m²/s³)")
axs[0].set_yscale('log')
axs[0].grid(True)

# (b) Wavelength
axs[1].plot(times, max_Lr, marker='o', color='tab:orange')
axs[1].set_title("(b) Horizontal Length Scale (Lmax)")
axs[1].set_ylabel("Lmax (km)")
axs[1].grid(True)

# (c) Depth at min
axs[2].plot(times, max_depths, marker='o', color='tab:green')
axs[2].set_title("(c) Depth at Spectral Min (Dmax)")
axs[2].set_ylabel("Depth (m)")
axs[2].set_xlabel("Time")
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.savefig(f"{figdir}/summary_timeseries.png", dpi=150)
plt.close()
print("✅ Saved summary_timeseries.png")

# ========== Convert PNGs to MP4 ==========

mp4_path = os.path.join(figdir, "movie-wb_spectra_7d_rolling.mp4")
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/wb_cross-spectrum_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {mp4_path}")
print(f"🎬 Saved movie to {mp4_path}")

# ========== Plot MLD-averaged spectrum ==========
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, mean_spec_in_mld, marker='o', color='tab:purple')
ax.set_title('Mean $w$-$b$ Spectrum in Mixed Layer')
ax.set_ylabel('spec_vp (m²/s³)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/mean_spec_in_MLD_timeseries.png", dpi=150)
plt.close()

# Submesoscale portion
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, mean_spec_in_mld_submeso, marker='o', color='tab:purple')
ax.set_title('Mean $w$-$b$ Spectrum in MLD (Submesoscale)')
ax.set_ylabel('spec_vp (m²/s³)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/mean_spec_in_MLD_submeso_timeseries.png", dpi=150)
plt.close()

print("✅ All plots completed.")

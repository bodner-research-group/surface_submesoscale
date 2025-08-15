##### Plot wb_cros of each week, compute wbmin, Lmax, Dmax
#####
##### wb_cros (variance-perserving cross-spectrum of vertical velocity and buoyancy)
##### wbmin (the minimum of wb_cros)
##### Lmax (the horizontal length scale corresponds to wbmin)
##### Dmax (the depth corresponds to wbmin)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from glob import glob
import pandas as pd
from set_colormaps import WhiteBlueGreenYellowRed

cmap = WhiteBlueGreenYellowRed()

from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# Input/output
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_cross_spectra_weekly"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_spectra_weekly_24hfilter"
out_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_max_spec_vp_filtered.nc"
os.makedirs(figdir, exist_ok=True)

# Global font size setting for figures
plt.rcParams.update({'font.size': 15})

# Filtering threshold (1/500 cpkm)
# kr_cutoff = 2 / 500  # cycles per km
kr_cutoff = 8 / 500

# Lists to collect results
times = []
max_values = []
max_depths = []
max_krs = []
max_Lr = []

def plot_wb_spectrum(nc_path):
    fname = os.path.basename(nc_path)
    date_str = fname.split("_")[-1].replace(".nc", "")
    time_val = pd.to_datetime(date_str)

    # Load data
    ds = xr.open_dataset(nc_path)
    k_r = ds.k_r
    depth = ds.depth
    spec_vp_real = ds.spec_vp_real

    # Filter low-wavenumber
    k_r_filtered = k_r.where(k_r >= kr_cutoff, drop=True)
    spec_vp_filtered = spec_vp_real.where(k_r >= kr_cutoff, drop=True)

    # Filter depth: -500m~0 (broadcast to 2D mask)
    valid_depth_mask = (depth >= -500) & (depth <= 0)
    spec_sel = spec_vp_filtered.where(valid_depth_mask, drop=True)
    
    # Find max value and its location
    max_val = spec_sel.max()
    idx = spec_sel.argmax(dim=["k", "freq_r"])
    max_k_idx = idx["k"].item()
    max_f_idx = idx["freq_r"].item()

    max_depth = depth[max_k_idx].item()
    max_k_r = k_r_filtered[max_f_idx].item()
    max_L_r = 1/max_k_r

    # Append results for NetCDF
    times.append(time_val)
    max_values.append(max_val.item())
    max_depths.append(max_depth)
    max_krs.append(max_k_r)
    max_Lr.append(max_L_r)

    # Plot
    plt.figure(figsize=(8, 6))
    # vmax = np.abs(spec_vp_filtered).max().item()
    vmax = 5e-10
    # norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    norm = TwoSlopeNorm(vcenter=vmax/2, vmin=0, vmax=vmax)

    pc = plt.pcolormesh(
        k_r_filtered, depth, spec_vp_filtered,
        shading='auto', cmap=cmap, norm=norm
    )
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.gca().invert_yaxis()  # Flip y-axis so depth increases downward
    plt.xlabel(r'Wavenumber $k_r$ (cpkm)')
    plt.ylabel('Depth (m)')
    plt.title(f'$w$-$b$ Cross-Spectrum (VP), {date_str}')
    plt.colorbar(pc, label=r'Spectral density (m$^2$s$^{-3}$)')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(f"{figdir}/wb_cross-spectrum_{date_str}.png", dpi=150)
    plt.close()
    print(f"Saved figure for {date_str}")

# ===== Loop through all .nc files and plot =====
all_spec_files = sorted(glob(os.path.join(input_dir, "wb_cross_spec_vp_real_24hfilter_*.nc")))
for f in all_spec_files:
    plot_wb_spectrum(f)

# ===== Save results to NetCDF =====
result_ds = xr.Dataset(
    {
        "max_spec_vp": (["time"], max_values),
        "depth_at_max": (["time"], max_depths),
        "kr_at_max": (["time"], max_krs),
        "Lr_at_max": (["time"], max_Lr),
    },
    coords={
        "time": pd.to_datetime(times)
    }
)

result_ds.to_netcdf(out_nc_path)
print(f"\nSaved summary NetCDF to: {out_nc_path}")


import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 16})

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# (a) Max spec_vp
axs[0].plot(times, max_values, marker='o', linestyle='-', color='tab:blue', label='Max spec_vp')
axs[0].set_yscale('log')
axs[0].set_ylabel('Max spec_vp (m²/s³)')
axs[0].set_title('(a) Peak Spectral Value over Time')
axs[0].grid(True, linestyle='--', which='major')  
axs[0].minorticks_on()  
axs[0].grid(True, linestyle=':', linewidth=0.5, which='minor')  
axs[0].legend()

# (b) Wavelength at Max
axs[1].plot(times, max_Lr, marker='o', linestyle='-', color='tab:orange', label='Wavelength at Max')
axs[1].set_ylabel('Wavelength (km)')
axs[1].set_title('(b) Wavelength at Spectral Peak')
axs[1].set_yticks([10, 20, 30, 40, 50])  
axs[1].grid(True, linestyle='--')
axs[1].legend()


# (c) Depth at Max
axs[2].plot(times, max_depths, marker='o', linestyle='-', color='tab:green', label='Depth at Max')
axs[2].set_ylabel('Depth (m)')
axs[2].set_title('(c) Depth at Spectral Peak')
axs[2].grid(True, linestyle='--')
axs[2].legend()
axs[2].set_xlabel('Time')

axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(figdir, "combined_spec_timeseries.png"), dpi=150)
plt.close()
print("Saved: combined_spec_timeseries.png")


##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_spectra_weekly_24hfilter"
# high-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-wb_spectra_weekly_24hfilter.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/wb_cross-spectrum_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

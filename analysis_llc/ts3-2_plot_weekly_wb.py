import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from glob import glob
import pandas as pd
from my_colormaps import WhiteBlueGreenYellowRed

cmap = WhiteBlueGreenYellowRed()

# Input/output
input_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/wb_cross_spectra_weekly"
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/wb_spectra_weekly"
out_nc_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/wb_max_spec_vp_filtered.nc"
os.makedirs(figdir, exist_ok=True)

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
    plt.title(f'$w$-$b$ Cross-Spectrum (VP) — {date_str}')
    plt.colorbar(pc, label=r'Spectral density (m$^2$s$^{-3}$)')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(f"{figdir}/wb_cross-spectrum_{date_str}.png", dpi=150)
    plt.close()
    print(f"Saved figure for {date_str}")

# ===== Loop through all .nc files and plot =====
all_spec_files = sorted(glob(os.path.join(input_dir, "wb_cross_spec_vp_real_*.nc")))
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

# figure dir
timeseries_figdir = os.path.join(figdir, "summary_timeseries")
os.makedirs(timeseries_figdir, exist_ok=True)

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
plot_timeseries("max_spec_vp", max_values, "Max spec_vp (m²/s³)", "max_spec_vp_timeseries.png", yscale='log')
plot_timeseries("depth_at_max", max_depths, "Depth at Max (m)", "depth_at_max_timeseries.png")
plot_timeseries("kr_at_max", max_krs, "k_r at Max (cpkm)", "kr_at_max_timeseries.png")
plot_timeseries("Lr_at_max", max_Lr, "Wavelength at peak spec (cpkm)", "Lr_at_max_timeseries.png")
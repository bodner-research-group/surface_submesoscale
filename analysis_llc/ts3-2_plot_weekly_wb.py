import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from glob import glob


# Input/output
input_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/wb_cross_spectra_weekly"
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/wb_spectra_weekly"
os.makedirs(figdir, exist_ok=True)

# Filtering threshold (1/500 cpkm)
kr_cutoff = 2 / 500  # cycles per km

def plot_wb_spectrum(nc_path):
    fname = os.path.basename(nc_path)
    date_str = fname.split("_")[-1].replace(".nc", "")

    # Load data
    ds = xr.open_dataset(nc_path)
    k_r = ds.k_r
    depth = ds.depth
    spec_vp_real = ds.spec_vp_real

    # Filter low-wavenumber
    k_r_filtered = k_r.where(k_r >= kr_cutoff, drop=True)
    spec_vp_filtered = spec_vp_real.where(k_r >= kr_cutoff, drop=True)

    # Plot
    plt.figure(figsize=(8, 6))
    # vmax = np.abs(spec_vp_filtered).max().item()
    vmax = 4e-10
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    pc = plt.pcolormesh(
        k_r_filtered, depth, spec_vp_filtered,
        shading='auto', cmap='RdBu', norm=norm
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



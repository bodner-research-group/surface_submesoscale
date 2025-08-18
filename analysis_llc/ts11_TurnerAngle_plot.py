import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob

from set_constant import domain_name, face, i, j  

# =====================
# CONFIGURATION
# =====================
turner_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle"
os.makedirs(figdir, exist_ok=True)


# =====================
# UTILITY FUNCTION TO PLOT A SINGLE WEEK
# =====================
def plot_turner_pdf(ncfile, save_dir):
    ds = xr.open_dataset(ncfile)

    beta_surf = ds["beta_surf"].data
    alpha_surf = ds["alpha_surf"].data
    pdf_values_h = ds["pdf_values_h"].data
    pdf_values_v = ds["pdf_values_v"].data
    x_grid = ds["x_grid"].data

    beta = np.nanmean(beta_surf)
    alpha = np.nanmean(alpha_surf)
    slope_rho = 1  # or beta / alpha

    v_cross = np.array([-slope_rho, 1.0])
    v_iso = np.array([1.0, slope_rho])
    v_cross /= np.linalg.norm(v_cross)
    v_iso /= np.linalg.norm(v_iso)

    # Setup plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Kernel PDF of Turner Angle")
    ax.set_xlabel(r"$\beta \partial S$")
    ax.set_ylabel(r"$\alpha \partial \theta$")
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_facecolor('white')

    scale = 3e-7  # domain-dependent
    x_all, y_all = [], []

    # Plot PDF vectors
    for n in range(len(x_grid)):
        angle_deg = x_grid[n]
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso

        # Horizontal PDF
        mag_h = pdf_values_h[n] * scale * 3
        xh, yh = [0, mag_h * dir_vec[0]], [0, mag_h * dir_vec[1]]
        ax.plot(xh, yh, color='orange', linewidth=0.7)
        x_all.append(xh[1])
        y_all.append(yh[1])

        # Vertical PDF
        mag_v = pdf_values_v[n] * scale
        xv, yv = [0, mag_v * dir_vec[0]], [0, mag_v * dir_vec[1]]
        ax.plot(xv, yv, color='green', linewidth=0.7)
        x_all.append(xv[1])
        y_all.append(yv[1])

    # Set axis limits
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    padding = 0.02
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
    ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])

    # Add zero lines
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)

    # Add vector arrows
    origin = [0, 0]
    ax.quiver(*origin, *(v_cross * scale / 50), color='red', angles='xy', scale_units='xy', scale=1, label='⊥ isopycnal', linewidth=2)
    ax.quiver(*origin, *(v_iso * scale / 50), color='blue', angles='xy', scale_units='xy', scale=1, label='∥ isopycnal', linewidth=2)

    # Legend
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    # Save figure
    date_tag = os.path.basename(ncfile).split("_")[-1].replace(".nc", "")
    fig_path = os.path.join(save_dir, f"Tu_TS_{date_tag}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {fig_path}")

# =====================
# LOOP OVER ALL NETCDF FILES
# =====================
nc_files = sorted(glob(os.path.join(turner_dir, "TuVH_7d_*.nc")))
print(f"Found {len(nc_files)} NetCDF files to plot.")

for ncfile in nc_files:
    plot_turner_pdf(ncfile, figdir)
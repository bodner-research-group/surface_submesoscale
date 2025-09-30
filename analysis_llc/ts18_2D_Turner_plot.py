import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
from glob import glob
from scipy.io import savemat
from scipy.interpolate import interp1d
from dask import delayed, compute
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j
from set_colormaps import WhiteBlueGreenYellowRed


# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("✅ Dask cluster started")
print("Dask dashboard:", client.dashboard_link)

# =====================
# Define constants and directories
# =====================
cmap = WhiteBlueGreenYellowRed()
# Global font size setting for figures
plt.rcParams.update({'font.size': 16})


nc_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_3D"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
os.makedirs(figdir, exist_ok=True)

nc_files = sorted(glob(os.path.join(nc_dir, "Turner_3D_7d_*.nc")))

# =====================
# Step 1: Compute global limits
# =====================
def compute_global_limits(nc_files):
    global_vlims = {
        "deta": [np.inf, -np.inf],
        "dt": [np.inf, -np.inf],
        "ds": [np.inf, -np.inf],
        "TuV": [180, -180],
        "TuH": [180, -180],
        "Tu_diff": [np.inf, -np.inf],
    }

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        deta = ds['deta_cross'][:-1, :-1]
        dt = ds['dt_cross'][:-1, :-1]
        ds_ = ds['ds_cross'][:-1, :-1]
        TuV = ds['TuV_deg'][:-1, :-1]
        TuH = ds['TuH_deg'][:-1, :-1]
        Tu_diff = TuV - TuH

        for key, data in zip(['deta', 'dt', 'ds', 'TuV', 'TuH', 'Tu_diff'], 
                             [deta, dt, ds_, TuV, TuH, Tu_diff]):
            data_vals = data.values
            finite_vals = data_vals[np.isfinite(data_vals)]
            if finite_vals.size == 0:
                continue
            vmax = finite_vals.max()
            vmin = -vmax
            global_vlims[key][0] = min(global_vlims[key][0], vmin)
            global_vlims[key][1] = max(global_vlims[key][1], vmax)

        ds.close()

    return global_vlims

print("🔍 Scanning all files for global limits...")
global_vlims = compute_global_limits(nc_files)
print("✅ Global limits computed.")

# =====================
# Step 2: Define processing function
# =====================
@delayed
def process_week(nc_file, vlims):
    date_tag = os.path.basename(nc_file).split("_")[-1].replace(".nc", "")
    print(f"🔄 Processing: {date_tag}")
    
    ds = xr.open_dataset(nc_file)

    deta_cross = ds['deta_cross']
    dt_cross = ds['dt_cross']
    ds_cross = ds['ds_cross']
    TuV_deg = ds['TuV_deg']
    TuH_deg = ds['TuH_deg']
    alpha = ds['alpha_surf']
    beta = ds['beta_surf']
    lon = ds['lon2d']
    lat = ds['lat2d']

    mask = np.isfinite(deta_cross) & np.isfinite(dt_cross) & np.isfinite(ds_cross)

    # --------------------------------
    # (3) 2D Scatter: Color-coded deta_cross
    # --------------------------------
    x = (beta * ds_cross).values[mask]
    y = (alpha * dt_cross).values[mask]
    z = np.abs(deta_cross.values[mask])

    # Sort by z (ascending)
    sort_idx = np.argsort(z)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    z_sorted = z[sort_idx]

    # Define color limits
    zmin, zmax = 0, 1e-5

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal', adjustable='box')  # Make x and y axis scale equal
    # sc = ax.scatter(x, y, c=z, cmap=cmocean.cm.balance, s=2, alpha=0.7, vmin=zmin, vmax=zmax)
    sc = ax.scatter(x_sorted, y_sorted, c=z_sorted, cmap=cmap, s=2, alpha=0.7, vmin=zmin, vmax=zmax)

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    
    ax.set_xlabel(r"$\beta \cdot \partial S_{cross}$")
    ax.set_ylabel(r"$\alpha \cdot \partial \theta_{cross}$")
    ax.set_xlim(-2e-8, 4e-8)
    ax.set_ylim(-3.5e-8, 1.5e-8)
    ax.set_title(f"Color: Cross-isopycnal SSH Gradient magnitude - {date_tag}")

    cbar = fig.colorbar(sc, ax=ax, label=r"$\vert\partial \eta_{cross}\vert$", shrink=0.8)

    scale = 3e-7

    slope_rho = 1  # Or slope_rho = beta / alpha if dynamic
    # Unit vectors
    v_cross = np.array([-slope_rho, 1.0])
    v_iso = np.array([1.0, slope_rho])
    v_cross /= np.linalg.norm(v_cross)
    v_iso /= np.linalg.norm(v_iso)

    # --- Add isopycnal lines ---
    min_ds = -1e-08
    max_ds = 1e-08
    min_dt = -1e-08
    max_dt = 1e-08

    S_line = np.linspace(min_ds * 7, max_ds * 7, 400)
    c_values = np.linspace(min_dt * 10, max_dt * 10, 100)

    for c in c_values:
        T_line = slope_rho * S_line + c
        ax.plot(S_line, T_line, '-', color='gray', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"2d_scatter_deta_cross_{date_tag}.png"), dpi=300)
    plt.close()

    # -----------------------------
    # (6) 2D Plot: PDF Vectors Colored by |deta|
    # -----------------------------

    ds_pdf = xr.open_dataset(nc_file)

    x_grid = ds_pdf["x_grid"].data
    pdf_values_h = ds_pdf["pdf_values_h"].data
    pdf_values_v = ds_pdf["pdf_values_v"].data
    alpha_val = np.nanmean(ds_pdf["alpha_surf"].data)
    beta_val = np.nanmean(ds_pdf["beta_surf"].data)

    # Compute mean |deta| in Turner angle bins
    TuH_vals = TuH_deg.values[mask]
    deta_vals = np.abs(z)
    bins = x_grid
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(TuH_vals, bins) - 1
    mean_deta_per_bin = np.array([
        deta_vals[digitized == i].mean() if np.any(digitized == i) else np.nan
        for i in range(len(bin_centers))
    ])
    interp_func = interp1d(
        bin_centers[np.isfinite(mean_deta_per_bin)],
        mean_deta_per_bin[np.isfinite(mean_deta_per_bin)],
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    z_vals = interp_func(x_grid)

    # Set color limits
    z_min_plot = 0
    z_max_plot = 3e-6
    normed_vals = np.clip(z_vals, z_min_plot, z_max_plot)
    normed_vals = (normed_vals - z_min_plot) / (z_max_plot - z_min_plot)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_title(f"Kernel PDF of Turner Angles\nWeek: {date_tag}", fontsize=16)
    ax.set_xlabel(r"$\beta \partial S$")
    ax.set_ylabel(r"$\alpha \partial \theta$")
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # --- Add isopycnal lines ---
    for c in c_values:
        T_line = slope_rho * S_line + c
        ax.plot(S_line, T_line, '-', color='gray', linewidth=0.5, alpha=0.3)

    # Plot PDF vectors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min_plot, vmax=z_max_plot))
    for angle_deg, mag_h, mag_v, color_val in zip(x_grid, pdf_values_h, pdf_values_v, normed_vals):
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso

        # Horizontal (colored by |deta|)
        dx_h = mag_h * dir_vec[0] * scale * 3
        dy_h = mag_h * dir_vec[1] * scale * 3
        ax.plot([0, dx_h], [0, dy_h], color=cmap(color_val), linewidth=1.0)

        # Vertical (green)
        dx_v = mag_v * dir_vec[0] * scale
        dy_v = mag_v * dir_vec[1] * scale
        ax.plot([0, dx_v], [0, dy_v], color='magenta', linewidth=0.7, alpha=0.7, linestyle=':')


    # Axis limits from global settings
    # (OPTIONAL: Use get_global_axis_limits if needed)
    ax.set_xlim([-1e-8, 3.5e-8])
    ax.set_ylim([-0.7e-8, 5.7e-8])

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)

    # Unit direction vectors
    origin = [0, 0]
    ax.quiver(*origin, *(v_cross * scale / 50), color='red', angles='xy', scale_units='xy', scale=1,
              label='⊥ isopycnal', linewidth=2)
    ax.quiver(*origin, *(v_iso * scale / 50), color='blue', angles='xy', scale_units='xy', scale=1,
              label='∥ isopycnal', linewidth=2)

    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    # Colorbar
    cbar = fig.colorbar(sm, ax=ax, label=r"Mean $|\partial \eta|$", shrink=0.8)

    plt.tight_layout()
    fig_path = os.path.join(figdir, f"2d_turner_pdf_with_color_{date_tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    ds_pdf.close()
    print(f"✅ Saved: {fig_path}")

    return fig_path

# =====================
# Step 3: Run in parallel with consistent limits
# =====================
tasks = [process_week(nc_file, global_vlims) for nc_file in nc_files]
results = compute(*tasks)

print("\n🎉 All weeks processed.")
for r in results:
    print(r)



import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-2d_scatter_deta_cross.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/2d_scatter_deta_cross_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")


import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-2d_turner_pdf_with_color.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/2d_turner_pdf_with_color_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

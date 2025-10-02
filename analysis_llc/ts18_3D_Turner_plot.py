import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import savemat
from scipy.interpolate import interp1d
from dask import delayed, compute
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j
from set_colormaps import WhiteBlueGreenYellowRed
import cmocean

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
plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

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
        "Tu_diff_abs": [0, 180],
        "Tu_diff_abs_new": [0, 180],
    }

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        deta = ds['deta_cross'][:-1, :-1]
        dt = ds['dt_cross'][:-1, :-1]
        ds_ = ds['ds_cross'][:-1, :-1]
        TuV = ds['TuV_deg'][:-1, :-1]
        TuH = ds['TuH_deg'][:-1, :-1]
        Tu_diff_abs = np.abs(TuV - TuH)
        Tu_diff_abs_new = np.abs(TuV - np.abs(TuH))

        for key, data in zip(['deta', 'dt', 'ds', 'TuV', 'TuH', 'Tu_diff_abs','Tu_diff_abs_new'], 
                             [deta, dt, ds_, TuV, TuH, Tu_diff_abs,Tu_diff_abs_new]):
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

    # --- Load data ---
    deta_cross = ds['deta_cross']
    dt_cross = ds['dt_cross']
    ds_cross = ds['ds_cross']
    TuV_deg = ds['TuV_deg']
    TuH_deg = ds['TuH_deg']
    alpha = ds['alpha_surf']
    beta = ds['beta_surf']
    lon = ds['lon2d']
    lat = ds['lat2d']
    x_grid = ds["x_grid"].data
    pdf_values_h = ds["pdf_values_h"].data
    pdf_values_v = ds["pdf_values_v"].data

    lon_plot = lon.transpose("j", "i").values[:-1, :-1]
    lat_plot = lat.transpose("j", "i").values[:-1, :-1]

    # --- Add isopycnal lines ---
    slope_rho = 1  # Or slope_rho = beta / alpha if dynamic
    v_cross = np.array([-slope_rho, 1.0])
    v_iso = np.array([1.0, slope_rho])
    v_cross /= np.linalg.norm(v_cross)
    v_iso /= np.linalg.norm(v_iso)

    min_ds = -0.5
    max_ds = 0.5
    min_dt = -0.5
    max_dt = 0.5
    S_line = np.linspace(min_ds, max_ds, 100)
    c_values = np.linspace(min_dt, max_dt, 100)

    mask = np.isfinite(deta_cross) & np.isfinite(dt_cross) & np.isfinite(ds_cross)

    # --- x, y, and z values for the 3D scatter plot ---
    x = (beta * ds_cross).values[mask]
    y = (alpha * dt_cross).values[mask]
    z = np.abs(deta_cross.values[mask])   # z = deta_cross.values[mask]
    # Sort by z (ascending)
    sort_idx = np.argsort(z)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    z_sorted = z[sort_idx]

    # --- Compute mean |deta| in Turner angle bins ---
    TuH_vals = TuH_deg.values[mask]
    deta_vals = np.abs(z)
    bins = x_grid # bins = np.linspace(-180, 180, 361)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(TuH_vals, bins) - 1
    mean_deta_per_bin = np.array([
        deta_vals[digitized == i].mean() if np.any(digitized == i) else np.nan
        for i in range(len(bin_centers))
    ])

    # --- Turner Angles ---
    valid = ~np.isnan(mean_deta_per_bin)
    interp_func = interp1d(bin_centers[valid], mean_deta_per_bin[valid], kind='linear', bounds_error=False, fill_value=np.nan)
    z_vals = interp_func(x_grid)

    x_proj, y_proj = [], []
    for angle_deg, mag_h in zip(x_grid, pdf_values_h):
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
        dx = mag_h * dir_vec[0] 
        dy = mag_h * dir_vec[1] 
        x_proj.append(dx)
        y_proj.append(dy)

    # --------------------------------
    # (1) Map: deta_cross, dt_cross, ds_cross
    # --------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)

    deta_plot = deta_cross[:-1, :-1]
    dt_plot = dt_cross[:-1, :-1]
    ds_plot = ds_cross[:-1, :-1]

    titles = ['SSH Gradient', 'Temperature Gradient', 'Salinity Gradient']
    data = [deta_plot, dt_plot, ds_plot]
    keys = ['deta', 'dt', 'ds']

    for i, ax in enumerate(axs):
        vmin, vmax = vlims[keys[i]]
        pcm = ax.pcolormesh(lon_plot, lat_plot, data[i], cmap='RdBu_r', shading='auto',
                            vmin=vmin, vmax=vmax)
        ax.set_title(f"{titles[i]} ({date_tag})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(pcm, ax=ax, orientation='vertical')

    plt.savefig(os.path.join(figdir, f"cross_gradients_map_{date_tag}.png"), dpi=300)
    plt.close()

    # --------------------------------
    # (2) Map: TuV_deg, TuH_deg, Difference
    # --------------------------------
    TuV_plot = TuV_deg[:-1, :-1]
    TuH_plot = TuH_deg[:-1, :-1]
    # Tu_diff_plot = TuV_plot - TuH_plot
    Tu_diff_abs_plot = np.abs(TuV_plot - np.abs(TuH_plot))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    angles = [TuV_plot, TuH_plot, Tu_diff_abs_plot]
    cmaps = ['twilight', 'twilight', cmap]
    titles = [f"TuV ({date_tag})", f"TuH ({date_tag})", f"|TuV - |TuH|| ({date_tag})"]

    vlims_diff = vlims["Tu_diff_abs"]
    diff_vmax = max(abs(vlims_diff[0]), abs(vlims_diff[1]))

    vmins = [vlims["TuV"][0], vlims["TuH"][0], 0]
    vmaxs = [vlims["TuV"][1], vlims["TuH"][1], diff_vmax]

    for i, ax in enumerate(axs):
        pcm = ax.pcolormesh(lon_plot, lat_plot, angles[i], cmap=cmaps[i], shading='auto',
                            vmin=vmins[i], vmax=vmaxs[i])
        ax.set_title(titles[i])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(pcm, ax=ax, orientation='vertical')

    plt.savefig(os.path.join(figdir, f"turner_angles_map_{date_tag}.png"), dpi=300)
    plt.close()

    # --------------------------------
    # (3) 2D Scatter: Color-coded deta_cross
    # --------------------------------
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

    # --- Add isopycnal lines ---
    for c in c_values:
        T_line = slope_rho * S_line + c
        ax.plot(S_line, T_line, '-', color='gray', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"2d_scatter_deta_cross_{date_tag}.png"), dpi=300)
    plt.close()

    # --------------------------------
    # (4) 3D Scatter
    # --------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define z (and color) limits
    zmin, zmax = 0, 8e-6

    # sc = ax.scatter(x, y, z, c=z, cmap=cmocean.cm.balance, alpha=0.6, s=1, vmin=zmin, vmax=zmax)
    sc = ax.scatter(x_sorted, y_sorted, z_sorted, c=z_sorted, cmap=cmap, alpha=0.6, s=1, vmin=zmin, vmax=zmax)

    ax.set_xlabel('β·dS_cross')
    ax.set_ylabel('α·dT_cross')
    ax.set_zlabel(r"$\vert\partial \eta_{cross}\vert$")
    ax.set_zlim(zmin, zmax)
    ax.set_xlim(-2e-8, 4e-8)
    ax.set_ylim(-3.5e-8, 1.5e-8)

    fig.colorbar(sc, label=r"$\vert\partial \eta_{cross}\vert$")

    ax.set_title(f"3D Scatter: Cross-Isopycnal SSH Gradient ({date_tag})")
    plt.savefig(os.path.join(figdir, f"3d_scatter_deta_cross_{date_tag}.png"), dpi=300)
    plt.close()
    # # savemat(os.path.join(figdir, f"3d_scatter_deta_cross_{date_tag}.mat"), {'x': x, 'y': y, 'z': z})
    
    # --------------------------------
    # (5) Histogram and KDE
    # --------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.histplot(z, kde=True, bins=361, color='skyblue', stat='density', edgecolor='none')
    ax.set_title(f"Histogram and KDE of SSH Gradient ({date_tag})")
    ax.set_xlabel(r'$\partial \eta$')
    ax.set_ylabel("Density")
    ax.set_ylim(0, 800000)
    ax.set_xlim(0, 7e-6)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.4)
    plt.savefig(os.path.join(figdir, f"hist_kde_deta_cross_{date_tag}.png"), dpi=300)
    plt.close()

    # -----------------------------
    # (6) Mean deta vs TuH
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_centers, mean_deta_per_bin, marker='o', linestyle='-')
    ax.set_ylim(0, 2.2e-6)
    ax.set_xlabel("Horizontal Turner Angle (deg)")
    ax.set_ylabel(r"Mean $|\partial \eta|$")
    ax.set_title(f"Mean SSH Gradient vs. Turner Angle ({date_tag})")
    ax.grid(True)
    plt.savefig(os.path.join(figdir, f"turner_angle_vs_deta_cross_{date_tag}.png"), dpi=300)
    plt.close()

    # -----------------------------
    # (7) 2D Plot: PDF Vectors Colored by |deta|
    # -----------------------------
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
        dx_h = mag_h * dir_vec[0] * 3
        dy_h = mag_h * dir_vec[1] * 3
        ax.plot([0, dx_h], [0, dy_h], color=cmap(color_val), linewidth=1.0)

        # Vertical (green)
        dx_v = mag_v * dir_vec[0]
        dy_v = mag_v * dir_vec[1]
        ax.plot([0, dx_v], [0, dy_v], color='magenta', linewidth=0.7, alpha=0.7, linestyle=':')


    # Axis limits from global settings
    # (OPTIONAL: Use get_global_axis_limits if needed)
    ax.set_xlim([-0.03, 0.12])
    ax.set_ylim([-0.025, 0.19])

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)

    # Unit direction vectors
    origin = [0, 0]
    ax.quiver(*origin, *(v_cross * 0.03), color='red', angles='xy', scale_units='xy', scale=1,
              label='⊥ isopycnal', linewidth=2)
    ax.quiver(*origin, *(v_iso * 0.03), color='blue', angles='xy', scale_units='xy', scale=1,
              label='∥ isopycnal', linewidth=2)

    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    # Colorbar
    cbar = fig.colorbar(sm, ax=ax, label=r"Mean $|\partial \eta|$", shrink=0.8)

    plt.tight_layout()
    fig_path = os.path.join(figdir, f"2d_turner_pdf_with_color_{date_tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # -----------------------------
    # (8) 3D Plot: PDF Vectors
    # -----------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    z_min_plot = 0
    z_max_plot = 3e-6
    normed_vals = np.clip(z_vals, z_min_plot, z_max_plot)
    normed_vals = (normed_vals - z_min_plot) / (z_max_plot - z_min_plot)

    for angle_deg, mag_h in zip(x_grid, pdf_values_h):
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
        dx = mag_h * dir_vec[0]  * 3
        dy = mag_h * dir_vec[1]  * 3
        ax.plot([0, dx], [0, dy], [0, 0], color='darkgray', linestyle='--', alpha=0.7, linewidth=0.7)

    for angle_deg, mag_v in zip(x_grid, pdf_values_v):
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
        dx = mag_v * dir_vec[0]
        dy = mag_v * dir_vec[1]
        ax.plot([0, dx], [0, dy], [0, 0], color='magenta', alpha=0.7, linewidth=0.7)

    for x, y, z, cval in zip(x_proj, y_proj, z_vals, normed_vals):
        if not np.isnan(z):
            ax.plot([0, x], [0, y], [0, z], color=cmap(cval), alpha=0.8, linewidth=1.0)

    sc = ax.scatter([0], [0], [0], c=[0], cmap=cmap, vmin=z_min_plot, vmax=z_max_plot)

    ax.set_xlabel(r"$\beta \, \partial S$")
    ax.set_ylabel(r"$\alpha \, \partial \theta$")
    ax.set_zlabel(r"Mean $|\partial \eta|$")
    ax.set_title(f"3D Turner PDF Vectors vs. SSH Gradient ({date_tag})")
    ax.set_xlim([-0.03, 0.12])
    ax.set_ylim([-0.025, 0.19])
    ax.set_zlim([0, z_max_plot])
    cbar = fig.colorbar(sc, ax=ax, label=r"Mean $|\partial \eta|$", shrink=0.5, fraction=0.02, pad=0.05)

    fig_path = os.path.join(figdir, f"3d_turner_pdf_with_lines_{date_tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    ds.close()

    print(f"✅ Finished: {date_tag}")
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
output_movie = f"{figdir}/movie-cross_gradients_map.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/cross_gradients_map_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-turner_angles_map.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/turner_angles_map_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-3d_scatter_deta_cross.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/3d_scatter_deta_cross_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-hist_kde_deta_cross.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/hist_kde_deta_cross_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-turner_angle_vs_deta_cross.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/turner_angle_vs_deta_cross_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-3d_turner_pdf_with_lines.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/3d_turner_pdf_with_lines_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-2d_scatter_deta_cross.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/2d_scatter_deta_cross_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
output_movie = f"{figdir}/movie-2d_turner_pdf_with_color.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/2d_turner_pdf_with_color_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

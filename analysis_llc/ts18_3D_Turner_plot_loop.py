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
plt.rcParams.update({'font.size': 14})

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
    }

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        deta = ds['deta_cross'][:-1, :-1]
        dt = ds['dt_cross'][:-1, :-1]
        ds_ = ds['ds_cross'][:-1, :-1]
        TuV = ds['TuV_deg'][:-1, :-1]
        TuH = ds['TuH_deg'][:-1, :-1]
        # Tu_diff = TuV - TuH
        Tu_diff_abs = np.abs(TuV - TuH)

        for key, data in zip(['deta', 'dt', 'ds', 'TuV', 'TuH', 'Tu_diff_abs'], 
                             [deta, dt, ds_, TuV, TuH, Tu_diff_abs]):
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
    # (1) Map: deta_cross, dt_cross, ds_cross
    # --------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)

    lon_plot = lon.transpose("j", "i").values[:-1, :-1]
    lat_plot = lat.transpose("j", "i").values[:-1, :-1]
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
    Tu_diff_abs_plot = np.abs(TuV_plot - TuH_plot)
    

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    angles = [TuV_plot, TuH_plot, Tu_diff_abs_plot]
    cmaps = ['twilight', 'twilight', cmap]
    titles = [f"TuV ({date_tag})", f"TuH ({date_tag})", f"|TuV - TuH| ({date_tag})"]

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

    # # --------------------------------
    # # (3) 3D Scatter
    # # --------------------------------
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # x = (beta * ds_cross).values[mask]
    # y = (alpha * dt_cross).values[mask]
    # z = deta_cross.values[mask]
    # sc = ax.scatter(x, y, z, c=z, cmap='RdBu', alpha=0.6, s=1)
    # ax.set_xlabel('β·dS_cross')
    # ax.set_ylabel('α·dT_cross')
    # ax.set_zlabel('deta_cross')
    # ax.set_zlim(-7e-6, 7e-6)
    # ax.set_xlim(-2e-8, 4e-8)
    # ax.set_ylim(-3.5e-8, 1.5e-8)
    # fig.colorbar(sc, label='deta_cross')
    # ax.set_title(f"3D Scatter: SSH Gradient ({date_tag})")
    # plt.savefig(os.path.join(figdir, f"3d_scatter_deta_cross_{date_tag}.png"), dpi=300)
    # plt.close()

    import cmocean

    # --------------------------------
    # (3) 3D Scatter
    # --------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = (beta * ds_cross).values[mask]
    y = (alpha * dt_cross).values[mask]
    # z = deta_cross.values[mask]
    z = np.abs(deta_cross.values[mask])

    # Define z (and color) limits
    zmin, zmax = 0, 8e-6

    # sc = ax.scatter(x, y, z, c=z, cmap=cmocean.cm.balance, alpha=0.6, s=1, vmin=zmin, vmax=zmax)
    sc = ax.scatter(x, y, z, c=z, cmap=cmap, alpha=0.6, s=1, vmin=zmin, vmax=zmax)

    ax.set_xlabel('β·dS_cross')
    ax.set_ylabel('α·dT_cross')
    ax.set_zlabel('|dη_cross|')
    ax.set_zlim(zmin, zmax)
    ax.set_xlim(-2e-8, 4e-8)
    ax.set_ylim(-3.5e-8, 1.5e-8)

    # fig.colorbar(sc, label='|dη_cross|')

    ax.set_title(f"3D Scatter: SSH Gradient ({date_tag})")
    plt.savefig(os.path.join(figdir, f"3d_scatter_deta_cross_{date_tag}.png"), dpi=300)
    plt.close()

    savemat(os.path.join(figdir, f"3d_scatter_deta_cross_{date_tag}.mat"), {'x': x, 'y': y, 'z': z})

    # --------------------------------
    # (4) Histogram and KDE
    # --------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.histplot(z, kde=True, bins=361, color='skyblue', stat='density', edgecolor='none')
    ax.set_title(f"Histogram and KDE of SSH Gradient ({date_tag})")
    ax.set_xlabel(r'$\partial \eta$')
    ax.set_ylabel("Density")
    ax.set_ylim(0, 350000)
    ax.set_xlim(-7e-6, 7e-6)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.4)
    plt.savefig(os.path.join(figdir, f"hist_kde_deta_cross_{date_tag}.png"), dpi=300)
    plt.close()

    # -----------------------------
    # (5) Mean deta vs TuH
    # -----------------------------
    TuH_vals = TuH_deg.values[mask]
    deta_vals = np.abs(z)
    bins = np.linspace(-180, 180, 361)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(TuH_vals, bins) - 1
    mean_deta_per_bin = np.array([
        deta_vals[digitized == i].mean() if np.any(digitized == i) else np.nan
        for i in range(len(bin_centers))
    ])

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
    # (6) 3D Plot: PDF Vectors
    # -----------------------------
    ds_pdf = xr.open_dataset(nc_file)

    x_grid = ds_pdf["x_grid"].data
    pdf_values_h = ds_pdf["pdf_values_h"].data
    pdf_values_v = ds_pdf["pdf_values_v"].data
    alpha_val = np.nanmean(ds_pdf["alpha_surf"].data)
    beta_val = np.nanmean(ds_pdf["beta_surf"].data)

    scale = 10000
    slope_rho = 1
    v_cross = np.array([-slope_rho, 1.0])
    v_iso = np.array([1.0, slope_rho])
    v_cross /= np.linalg.norm(v_cross)
    v_iso /= np.linalg.norm(v_iso)

    valid = ~np.isnan(mean_deta_per_bin)
    interp_func = interp1d(bin_centers[valid], mean_deta_per_bin[valid], kind='linear', bounds_error=False, fill_value=np.nan)
    z_vals = interp_func(x_grid)

    x_proj, y_proj = [], []
    for angle_deg, mag_h in zip(x_grid, pdf_values_h):
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
        dx = mag_h * dir_vec[0] * beta_val * scale * 20
        dy = mag_h * dir_vec[1] * alpha_val * scale * 20
        x_proj.append(dx)
        y_proj.append(dy)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    z_min_plot = 0
    z_max_plot = 3e-6
    normed_vals = np.clip(z_vals, z_min_plot, z_max_plot)
    normed_vals = (normed_vals - z_min_plot) / (z_max_plot - z_min_plot)

    for x, y, z, cval in zip(x_proj, y_proj, z_vals, normed_vals):
        if not np.isnan(z):
            ax.plot([0, x], [0, y], [0, z], color=cmap(cval), alpha=0.8, linewidth=1.0)

    sc = ax.scatter([0], [0], [0], c=[0], cmap=cmap, vmin=z_min_plot, vmax=z_max_plot)

    for angle_deg, mag_h in zip(x_grid, pdf_values_h):
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
        dx = mag_h * dir_vec[0] * beta_val * scale * 20
        dy = mag_h * dir_vec[1] * alpha_val * scale * 20
        ax.plot([0, dx], [0, dy], [0, 0], color='darkgray', linestyle='--', alpha=0.7, linewidth=0.7)

    for angle_deg, mag_v in zip(x_grid, pdf_values_v):
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
        dx = mag_v * dir_vec[0] * beta_val * scale * 10
        dy = mag_v * dir_vec[1] * alpha_val * scale * 10
        ax.plot([0, dx], [0, dy], [0, 0], color='magenta', alpha=0.7, linewidth=0.7)

    ax.set_xlabel(r"$\beta \, \partial S$")
    ax.set_ylabel(r"$\alpha \, \partial \theta$")
    ax.set_zlabel(r"Mean $|\partial \eta|$")
    ax.set_title(f"3D Turner PDF Vectors vs. SSH Gradient ({date_tag})")
    ax.set_xlim([-1, 3])
    ax.set_ylim([-0.5, 1])
    ax.set_zlim([0, z_max_plot])
    cbar = fig.colorbar(sc, ax=ax, label=r"Mean $|\partial \eta|$", shrink=0.5, fraction=0.02, pad=0.05)

    fig_path = os.path.join(figdir, f"3d_turner_pdf_with_lines_{date_tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    ds_pdf.close()

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

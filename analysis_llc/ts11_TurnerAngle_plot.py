import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
from set_constant import domain_name
from matplotlib.colors import to_rgba

# =====================
# CONFIGURATION
# =====================
turner_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle"
os.makedirs(figdir, exist_ok=True)

# Global font size setting for figures
plt.rcParams.update({'font.size': 13})

# =====================
# GET GLOBAL AXIS LIMITS
# =====================
def get_global_axis_limits(nc_files: list, scale: float = 3e-7):
    x_vals, y_vals = [], []

    for ncfile in nc_files:
        ds = xr.open_dataset(ncfile)
        beta = np.nanmean(ds["beta_surf"].data)
        alpha = np.nanmean(ds["alpha_surf"].data)
        slope_rho = 1

        v_cross = np.array([-slope_rho, 1.0])
        v_iso = np.array([1.0, slope_rho])
        v_cross /= np.linalg.norm(v_cross)
        v_iso /= np.linalg.norm(v_iso)

        pdf_values_h = ds["pdf_values_h"].data
        pdf_values_v = ds["pdf_values_v"].data
        x_grid = ds["x_grid"].data

        for n in range(len(x_grid)):
            angle_deg = x_grid[n]
            dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso

            mag_h = pdf_values_h[n] * scale * 3
            mag_v = pdf_values_v[n] * scale

            x_vals.extend([mag_h * dir_vec[0], mag_v * dir_vec[0]])
            y_vals.extend([mag_h * dir_vec[1], mag_v * dir_vec[1]])

        ds.close()

    padding = 0.02
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)

    x_range = x_max - x_min
    y_range = y_max - y_min

    return (
        x_min - padding * x_range,
        x_max + padding * x_range,
        y_min - padding * y_range,
        y_max + padding * y_range,
    )


# =====================
# PLOT ONE FILE
# =====================
def plot_turner_pdf(ncfile: str, save_dir: str, axis_limits: tuple, scale: float = 3e-7):
    ds = xr.open_dataset(ncfile)

    beta = np.nanmean(ds["beta_surf"].data)
    alpha = np.nanmean(ds["alpha_surf"].data)
    slope_rho = 1  # Or slope_rho = beta / alpha if dynamic

    # Unit vectors
    v_cross = np.array([-slope_rho, 1.0])
    v_iso = np.array([1.0, slope_rho])
    v_cross /= np.linalg.norm(v_cross)
    v_iso /= np.linalg.norm(v_iso)

    pdf_values_h = ds["pdf_values_h"].data
    pdf_values_v = ds["pdf_values_v"].data
    x_grid = ds["x_grid"].data

    date_tag = os.path.basename(ncfile).split("_")[-1].replace(".nc", "")

    # Setup plot
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_title(f"Kernel PDF of Turner Angles\nWeek: {date_tag}", fontsize=16)
    ax.set_xlabel(r"$\beta \partial S$")
    ax.set_ylabel(r"$\alpha \partial \theta$")
    ax.set_aspect("equal")
    ax.set_facecolor("white")

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

    # --- Plot PDF vectors ---
    for n in range(len(x_grid)):
        angle_deg = x_grid[n]
        dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso

        # Horizontal PDF
        mag_h = pdf_values_h[n] * scale * 3
        xh, yh = [0, mag_h * dir_vec[0]], [0, mag_h * dir_vec[1]]
        ax.plot(xh, yh, color='orange', linewidth=0.7, label=r"$\mathrm{Tu}_H$" if n == 0 else "")

        # Vertical PDF
        mag_v = pdf_values_v[n] * scale
        xv, yv = [0, mag_v * dir_vec[0]], [0, mag_v * dir_vec[1]]
        ax.plot(xv, yv, color='green', linewidth=0.7, label=r"$\mathrm{Tu}_V$" if n == 0 else "")

    # Axis limits
    ax.set_xlim([axis_limits[0], axis_limits[1]])
    ax.set_ylim([axis_limits[2], axis_limits[3]])

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

    # Save figure
    fig_path = os.path.join(save_dir, f"Tu_TS_{date_tag}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    ds.close()
    print(f"Saved: {fig_path}")


# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    nc_files = sorted(glob(os.path.join(turner_dir, "TuVH_7d_*.nc")))
    print(f"Found {len(nc_files)} NetCDF files to plot.")

    # Get consistent global axis limits
    global_limits = get_global_axis_limits(nc_files)
    print(f"Global axis limits: {global_limits}")

    for ncfile in nc_files:
        plot_turner_pdf(ncfile, figdir, axis_limits=global_limits)


##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle"
# high-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-TurnerAngle.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/Tu_TS_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")






######################################################################################
######## Quantify the agreement between horizontal and vertical Turner Angles ########
######################################################################################

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.dates as mdates
import re

from set_constant import domain_name

# Global font size setting for figures
plt.rcParams.update({'font.size': 14})

# ========== CONFIGURATION ==========
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle"
figdir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle"
os.makedirs(figdir, exist_ok=True)

file_pattern = os.path.join(data_dir, "TuVH_7d_*.nc")
nc_files = sorted(glob.glob(file_pattern))

# =======================
# 2. Initialize containers
# =======================
dates = []
TuV_means, TuH_means = [], []
Tu_diff_means, Tu_RMSEs, Tu_corrs, Tu_agreements = [], [], [], []

# =======================
# 3. Loop through each file
# =======================
for fpath in nc_files:
    filename = os.path.basename(fpath)

    # Try to extract date using flexible parsing
    date_str = filename.replace("TuVH_7d_", "").replace(".nc", "")
    try:
        date = pd.to_datetime(date_str)  # handles both 'YYYYMMDD' and 'YYYY-MM-DD'
    except ValueError:
        print(f"⚠️ Skipping unrecognized file: {fpath}")
        continue

    try:
        ds = xr.open_dataset(fpath)
        TuV = ds["TuV_deg"].values
        TuH = ds["TuH_deg"].values

        mask = ~np.isnan(TuV) & ~np.isnan(TuH)
        if np.sum(mask) < 100:
            print(f"Skipping {date_str}: insufficient valid data")
            continue

        TuV_flat = TuV[mask].flatten()
        TuH_flat = TuH[mask].flatten()

        # Compute statistics
        mean_TuV = np.nanmean(TuV_flat)
        mean_TuH = np.nanmean(TuH_flat)
        diff_mean = np.nanmean(np.abs(TuV_flat - TuH_flat))
        rmse = np.sqrt(np.nanmean((TuV_flat - TuH_flat) ** 2))
        agreement = np.mean(np.abs(TuV_flat - TuH_flat) <= 10)
        corr, _ = pearsonr(TuV_flat, TuH_flat)

        # Store
        dates.append(date)
        TuV_means.append(mean_TuV)
        TuH_means.append(mean_TuH)
        Tu_diff_means.append(diff_mean)
        Tu_RMSEs.append(rmse)
        Tu_agreements.append(agreement)
        Tu_corrs.append(corr)

        ds.close()

    except Exception as e:
        print(f"❌ Failed to process {fpath}: {e}")
        continue

# =======================
# 4. Plot time series
# =======================
if len(dates) == 0:
    print("No valid data files processed. Exiting.")
    exit()

dates = pd.to_datetime(dates)
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

Tu_agreements_pct = np.array(Tu_agreements) * 100
Tu_corrs_pct = np.array(Tu_corrs) * 100

# Plot mean Turner angles
ax[0].plot(dates, TuV_means, label="Mean TuV (Vertical)", marker="o",linewidth=2)
ax[0].plot(dates, TuH_means, label="Mean TuH (Horizontal)", marker="x",linewidth=2)
ax[0].set_ylabel("Turner Angle (°)")
ax[0].set_title("Mean Turner Angles Over Time")
ax[0].legend()
ax[0].grid(True)

# Plot agreement metrics
ax[1].plot(dates, Tu_diff_means, label="Mean Abs Difference", color="orange",linewidth=2)
ax[1].plot(dates, Tu_RMSEs, label="RMSE", color="green",linewidth=2)
ax[1].plot(dates, Tu_agreements_pct, label="% Agreement (±10°)", color="gray",linestyle="-.")
ax[1].plot(dates, Tu_corrs_pct, label="Correlation (%)", color="gray",linestyle=":")
ax[1].set_ylabel("Agreement Metric")
ax[1].set_xlabel("Date")
ax[1].set_title("TuV vs TuH Agreement Metrics")
ax[1].legend()
ax[1].grid(True)

# Format date axis
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()

# Save figure
fig_path = os.path.join(figdir, "TurnerAngle_Timeseries_Agreement.png")
fig.savefig(fig_path, dpi=300)
print(f"\n Saved figure to: {fig_path}")

# =======================
# 5. Save statistics to CSV
# =======================
df_stats = pd.DataFrame({
    "date": dates,
    "TuV_mean": TuV_means,
    "TuH_mean": TuH_means,
    "Mean_abs_diff": Tu_diff_means,
    "RMSE": Tu_RMSEs,
    "Pearson_corr": Tu_corrs,
    "Agreement_±10deg": Tu_agreements,
})

csv_out = os.path.join(data_dir, "TurnerAngle_Timeseries_Stats.csv")
df_stats.to_csv(csv_out, index=False)
print(f" Saved statistics to: {csv_out}")

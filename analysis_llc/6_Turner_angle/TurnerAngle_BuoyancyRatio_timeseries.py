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

domain_name = "icelandic_basin"

# Global font size setting for figures
plt.rcParams.update({'font.size': 14})

# ========== CONFIGURATION ==========
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_BuoyancyRatio_daily"
figdir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/TurnerAngle_BuoyancyRatio_daily"
os.makedirs(figdir, exist_ok=True)

file_pattern = os.path.join(data_dir, "TurnerAngle_BuoyancyRatio_daily_*.nc")
nc_files = sorted(glob.glob(file_pattern))

# =======================
# 2. Initialize containers
# =======================
dates = []
buoyancy_ratio_linearEOS_means, buoyancy_ratio_means = [], []
TuV_means, TuH_means = [], []
Tu_diff_means, Tu_diff_means_new, Tu_RMSEs, Tu_corrs, Tu_agreements, Tu_agreements_new = [], [], [], [], [], []

# =======================
# 3. Loop through each file
# =======================
for fpath in nc_files:
    filename = os.path.basename(fpath)

    # Try to extract date using flexible parsing
    date_str = filename.replace("TurnerAngle_BuoyancyRatio_daily_", "").replace(".nc", "")
    try:
        date = pd.to_datetime(date_str)  # handles both 'YYYYMMDD' and 'YYYY-MM-DD'
    except ValueError:
        print(f"⚠️ Skipping unrecognized file: {fpath}")
        continue

    try:
        ds = xr.open_dataset(fpath)
        TuV = ds["TuV_deg"].values
        TuH = ds["TuH_deg"].values
        buoyancy_ratio = ds["buoyancy_ratio"].values
        buoyancy_ratio_linearEOS = ds["buoyancy_ratio_linearEOS"].values

        mask = ~np.isnan(TuV) & ~np.isnan(TuH)
        if np.sum(mask) < 100:
            print(f"Skipping {date_str}: insufficient valid data")
            continue

        TuV_flat = TuV[mask].flatten()
        TuH_flat = TuH[mask].flatten()
        buoyancy_ratio_flat = buoyancy_ratio[mask].flatten()
        buoyancy_ratio_linearEOS_flat = buoyancy_ratio_linearEOS[mask].flatten()

        # Compute statistics
        mean_buoyancy_ratio = np.nanmean(buoyancy_ratio_flat)
        mean_buoyancy_ratio_linearEOS = np.nanmean(buoyancy_ratio_linearEOS_flat)

        mean_TuV = np.nanmean(TuV_flat)
        mean_TuH = np.nanmean(TuH_flat)
        diff_mean = np.nanmean(np.abs(TuV_flat - TuH_flat))
        diff_mean_new = np.nanmean(np.abs(TuV_flat - np.abs(TuH_flat)))
        rmse = np.sqrt(np.nanmean((TuV_flat - TuH_flat) ** 2))
        agreement = np.mean(np.abs(TuV_flat - TuH_flat) <= 10)
        agreement_new = np.mean(np.abs(TuV_flat - np.abs(TuH_flat)) <= 10)
        corr, _ = pearsonr(TuV_flat, TuH_flat)

        # Store
        dates.append(date)
        buoyancy_ratio_means.append(mean_buoyancy_ratio)
        buoyancy_ratio_linearEOS_means.append(mean_buoyancy_ratio_linearEOS)
        TuV_means.append(mean_TuV)
        TuH_means.append(mean_TuH)
        Tu_diff_means.append(diff_mean)
        Tu_diff_means_new.append(diff_mean_new)
        Tu_RMSEs.append(rmse)
        Tu_agreements.append(agreement)
        Tu_agreements_new.append(agreement_new)
        Tu_corrs.append(corr)

        ds.close()

    except Exception as e:
        print(f"❌ Failed to process {fpath}: {e}")
        continue



# =======================
# 4. Plot time series
# =======================
import matplotlib.pyplot as plt

if len(dates) == 0:
    print("No valid data files processed. Exiting.")
    exit()

dates = pd.to_datetime(dates)
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

Tu_agreements_pct = np.array(Tu_agreements) * 100
Tu_agreements_new_pct = np.array(Tu_agreements_new) * 100
Tu_corrs_pct = np.array(Tu_corrs) * 100

# Plot mean Turner angles
ax[0].plot(dates, TuV_means, label="Mean TuV (Vertical)", marker="o",linewidth=2)
ax[0].plot(dates, TuH_means, label="Mean TuH (Horizontal)", marker="x",linewidth=2)
ax[0].set_ylabel("Turner Angle (°)")
ax[0].set_title("Mean Turner Angles Over Time")
ax[0].legend()
ax[0].grid(True)

# Plot agreement metrics
ax[1].plot(dates, Tu_diff_means_new, label="|TuV-|TuH||", color="orange",linewidth=2)
ax[1].plot(dates, Tu_diff_means, label="|TuV-TuH|", color="orange",linestyle="--",linewidth=2)
ax[1].plot(dates, Tu_RMSEs, label="RMSE", color="green",linewidth=2)
ax[1].plot(dates, Tu_agreements_pct, label="% Agreement (±10°)", color="gray",linestyle="-.")
ax[1].plot(dates, Tu_agreements_new_pct, label="% Agreement (±10°), using |TuH|", color="gray",linestyle="--")
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
fig_path = os.path.join(figdir, "TurnerAngle_Timeseries_Agreement_daily.png")
fig.savefig(fig_path, dpi=300)
print(f"\n Saved figure to: {fig_path}")


# Create DataFrame
df_stats = pd.DataFrame({
    "date": dates,
    "buoyancy_ratio_means": buoyancy_ratio_means,
    "buoyancy_ratio_linearEOS_means": buoyancy_ratio_linearEOS_means,
    "TuV_means": TuV_means,
    "TuV_means": TuV_means,
    "TuH_means": TuH_means,
    "Tu_diff_means": Tu_diff_means,
    "Tu_diff_means_new": Tu_diff_means_new,
    "Tu_agreements_pct": Tu_agreements_pct,
    "Tu_agreements_new_pct": Tu_agreements_new_pct,
    "Tu_RMSEs": Tu_RMSEs,
    "Tu_corrs_pct": Tu_corrs_pct,
})

# Set the output directory
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"  # Change this to your actual output directory

# Convert DataFrame to xarray Dataset
ds_stats = xr.Dataset.from_dataframe(df_stats.set_index("date"))

# Save to NetCDF
nc_out = os.path.join(output_dir, "TurnerAngle_BuoyancyRatio_daily_Timeseries.nc")
ds_stats.to_netcdf(nc_out)

print(f"✅ Saved statistics to: {nc_out}")


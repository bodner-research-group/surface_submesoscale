import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import pandas as pd
import matplotlib.dates as mdates

from set_colormaps import WhiteBlueGreenYellowRed
from set_constant import domain_name

# ==========================
# Directories and files
# ==========================
cmap = WhiteBlueGreenYellowRed()

input_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_cross_spectra_weekly/wb_cross_spec_vp_real_7day_rolling.nc"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_spectra_weekly_7d_rolling"
out_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_max_spec_vp_filtered_7d_rolling_mean.nc"

os.makedirs(figdir, exist_ok=True)
plt.rcParams.update({'font.size': 15})

# ==========================
# Load MLD
# ==========================
f_Hml = f'/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_7d_rolling.nc'
Hml_mean = xr.open_dataset(f_Hml).Hml_mean  # (time,)

# ==========================
# Load wb spectrum
# ==========================
ds = xr.open_dataset(input_path)
spec_vp_real = ds.spec_vp_real  # (time, Z, freq_r)
depth = ds.depth
k_r = ds.k_r
times = ds.time.values

# ==========================
# Constants
# ==========================
kr_cutoff = 2 / 500      # cpkm (>500 km excluded)
kr_cutoff_meso = 1 / 30  # cpkm (<30 km for submesoscale)

# ==========================
# Prepare storage
# ==========================
max_values = []
max_depths = []
max_krs = []
max_Lr = []

mean_spec_in_mld = []
mean_spec_in_mld_submeso = []
vertical_bf_submeso_mld = []   # vertical buoyancy flux (submeso, in MLD)
energy_weighted_Lr = []        # energy-weighted mean wavelength


# ==========================
# Loop over time
# ==========================
# date_str = '2011-11-05'
# t_idx = 4
# time_val = '2011-11-05T00:00:00.000000000'

for t_idx, time_val in enumerate(times):
    date_str = np.datetime_as_string(time_val, unit='D')

    print(f"Processing {date_str}...")

    # 2D spectrum slice
    spec_2d = spec_vp_real.isel(time=t_idx)  # (Z, freq_r)
    depth_vals = depth.values

    # --- Subset by kr_cutoff ---
    kr_mask = k_r >= kr_cutoff
    spec_filtered = spec_2d[:, kr_mask]        
    k_r_filtered = k_r[kr_mask].values

    # --- Subset for submesoscale (<30 km) ---
    kr_mask_sub = k_r >= kr_cutoff_meso
    spec_submeso = spec_2d[:, kr_mask_sub]
    k_r_submeso = k_r[kr_mask_sub].values

    # --- Depth mask ---
    # valid_depth_mask = (depth >= Hml_mean.min().values) & (depth <= 0)
    valid_depth_mask = (depth >-600) & (depth <= 0)
    spec_filtered = spec_filtered.where(valid_depth_mask, drop=True)
    spec_submeso = spec_submeso.where(valid_depth_mask, drop=True)

    # --- MLD for this week ---
    # try:
    #     week_idx = np.where(Hml_mean.time.values == np.datetime64(time_val))[0][0]
    #     Hml_this_week = Hml_mean.isel(time=week_idx).item()
    # except IndexError:
    #     Hml_this_week = np.nan

    # Convert both arrays to dates only
    dates_only = Hml_mean.time.values.astype('datetime64[D]')
    time_val_date = np.datetime64(time_val, 'D')

    try:
        week_idx = np.where(dates_only == time_val_date)[0][0]
        Hml_this_week = Hml_mean.isel(time=week_idx).item()
    except IndexError:
        Hml_this_week = np.nan


    # --- Mean spectrum in MLD ---
    if not np.isnan(Hml_this_week):
        spec_in_mld = spec_filtered.where(depth >= Hml_this_week, drop=True)
        mean_val = spec_in_mld.mean().item()
    else:
        mean_val = np.nan
    mean_spec_in_mld.append(mean_val)

    # --- Submesoscale portion ---
    if not np.isnan(Hml_this_week):
        spec_submeso_mld = spec_submeso.where(depth >= Hml_this_week, drop=True)
        mean_val_submeso = spec_submeso_mld.mean().item()
    else:
        mean_val_submeso = np.nan
    mean_spec_in_mld_submeso.append(mean_val_submeso)

    # --- Minimum (most negative contribution) ---
    if spec_in_mld.size > 0:
        min_val = spec_in_mld.min().item()
        idx = np.unravel_index(np.argmin(spec_in_mld.values), spec_in_mld.shape)
        depth_at_min = spec_in_mld.Z[idx[0]].item()
        kr_at_min = k_r_filtered[idx[1]].item()
        Lr_at_min = 1 / kr_at_min
    else:
        min_val = np.nan
        depth_at_min = np.nan
        kr_at_min = np.nan
        Lr_at_min = np.nan

    max_values.append(min_val)
    max_depths.append(depth_at_min)
    max_krs.append(kr_at_min)
    max_Lr.append(Lr_at_min)

    # ==========================
    # --- Vertical buoyancy flux: submeso (<30 km) in MLD ---
    # ==========================
    if not np.isnan(Hml_this_week):
        spec_vals = spec_submeso_mld.fillna(0.0).values  # (Z, kr)
        # lambda_vals_sub = 1.0 / k_r_submeso
        # lambda_vals_sub = lambda_vals_sub[np.newaxis, :]  # broadcast to Z
        # vertical_bf = np.trapezoid(spec_vals * lambda_vals_sub, x=k_r_submeso, axis=-1)
        vertical_bf = np.trapezoid(spec_vals, x=k_r_submeso, axis=-1)
        vertical_bf = np.nanmean(vertical_bf)  # average in depth
    else:
        vertical_bf = np.nan
    vertical_bf_submeso_mld.append(vertical_bf)

    # ==========================
    # --- Variance-weighted mean wavelength ---
    # ==========================
    if spec_filtered.size > 0:
        spec_vals = spec_filtered.fillna(0.0).values
        lambda_vals = 1.0 / k_r_filtered
        lambda_vals = lambda_vals[np.newaxis, :]  # broadcast to Z
        numerator = np.trapezoid(spec_vals * lambda_vals, x=k_r_filtered, axis=-1)
        denominator = np.trapezoid(spec_vals, x=k_r_filtered, axis=-1)
        ew_Lr = np.nanmean(numerator / denominator)  # average in depth
    else:
        ew_Lr = np.nan
    energy_weighted_Lr.append(ew_Lr)

    # ==========================
    # --- Plot ---
    # ==========================
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = TwoSlopeNorm(vcenter=1e-10, vmin=0, vmax=5e-10)
    pc = ax.pcolormesh(
        k_r_filtered, -spec_filtered.Z, spec_filtered,
        shading='auto', cmap=cmap, norm=norm
    )
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_title(f'$w$-$b$ Cross-Spectrum (VP), {date_str}')
    ax.set_xlabel('Wavenumber $k_r$ (cpkm)')
    ax.set_ylabel('Depth (m)')

    # 30 km cutoff line at bottom
    ax.axvline(kr_cutoff_meso, color='r', linestyle='--')
    ax.text(kr_cutoff_meso, 0.05, '30 km cutoff', color='r', ha='center',
            va='bottom', rotation=90, transform=ax.get_xaxis_transform())

    fig.colorbar(pc, ax=ax, label=r'Spectral density (m$^2$s$^{-3}$)')
    plt.tight_layout()
    plt.savefig(f"{figdir}/wb_cross-spectrum_{date_str}.png", dpi=150)
    plt.close()




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





# ==========================
# Convert to pandas datetime
# ==========================
time_pd = pd.to_datetime(times)

# ==========================
# Plot: Most negative spectral value in MLD
# ==========================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_pd, max_values, marker='o', color='tab:red')
ax.set_title('Most Negative $w$-$b$ Spectrum Value in MLD')
ax.set_ylabel('spec_vp (m²/s³)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/min_spec_in_MLD_timeseries.png", dpi=150)
plt.close()

# ==========================
# Plot: Vertical buoyancy flux in submeso MLD
# ==========================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_pd, vertical_bf_submeso_mld, marker='o', color='tab:blue')
ax.set_title('Vertical Buoyancy Flux in MLD (Submesoscale)')
ax.set_ylabel('Flux (arbitrary units)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/vertical_bf_submeso_MLD_timeseries.png", dpi=150)
plt.close()

# ==========================
# Plot: Energy-weighted mean wavelength
# ==========================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_pd, energy_weighted_Lr, marker='o', color='tab:green')
ax.set_title('Energy-Weighted Mean Wavelength ($L_r$)')
ax.set_ylabel('Wavelength (km)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/energy_weighted_Lr_timeseries.png", dpi=150)
plt.close()


# ==========================
# Plot: Wavelength at Minimum Spectrum (Lr_at_min)
# ==========================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_pd, max_Lr, marker='o', color='tab:orange')
ax.set_title('Wavelength at Minimum $w$-$b$ Spectrum Value in MLD')
ax.set_ylabel('Wavelength $L_r$ (km)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/Lr_at_min_spec_MLD_timeseries.png", dpi=150)
plt.close()

print("📏 Lr_at_min time series plot saved.")



# ==========================
# Plot: Depth at Minimum Spectrum (depth_at_min)
# ==========================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_pd, max_depths, marker='o', color='tab:brown')
ax.set_title('Depth of Minimum $w$-$b$ Spectrum Value in MLD')
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Time')
ax.invert_yaxis()  # Optional: deeper values down
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/depth_at_min_spec_MLD_timeseries.png", dpi=150)
plt.close()

print("🌊 depth_at_min time series plot saved.")


print("✅ All plots completed.")

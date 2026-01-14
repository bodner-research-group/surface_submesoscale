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
kr_cutoff_meso = 1 / 20  # cpkm (<30 km for submesoscale)

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

# New: Store MLD-averaged spectra and peak wavenumber
mean_spec_mld_by_kr = []
kr_peak_list = []


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

    spec_2d = spec_2d.where(spec_2d >= 0, 0)

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
        spec_in_mld = spec_filtered.where((depth >= Hml_this_week) & (depth >= -600), drop=True)
        mean_val = spec_in_mld.mean().item()
    else:
        mean_val = np.nan
    mean_spec_in_mld.append(mean_val)

    # Compute mean spectrum over depth (in MLD), retain kr axis
    if not np.isnan(Hml_this_week):
        spec_mld_mean_by_kr = spec_in_mld.mean(dim='k')  # mean over depth
        mean_spec_mld_by_kr.append(spec_mld_mean_by_kr.values)
        
        # Find peak wavenumber
        kr_idx_peak = np.argmax(spec_mld_mean_by_kr.values)
        kr_peak = k_r.values[kr_idx_peak]
        kr_peak_list.append(kr_peak)
    else:
        mean_spec_mld_by_kr.append(np.full(k_r.shape, np.nan))
        kr_peak_list.append(np.nan)



    # --- Submesoscale portion ---
    if not np.isnan(Hml_this_week):
        spec_submeso_mld = spec_submeso.where((depth >= Hml_this_week) & (depth >= -600), drop=True)
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
    # if not np.isnan(Hml_this_week):
    #     # k_r_m = k_r_submeso * 1e-3   # convert from cpkm → cpm (m⁻¹)
    #     # lambda_vals_sub = 1.0 / k_r_m  # wavelength [m]
    #     # lambda_vals_sub = lambda_vals_sub[np.newaxis, :]  # broadcast to Z
    #     # Integration
    #     spec_vals = spec_submeso_mld.fillna(0.0).values  # (Z, kr)
    #     vertical_bf = np.trapezoid(spec_vals/k_r_submeso, x=k_r_submeso, axis=-1)
    #     # vertical_bf = np.trapezoid(spec_vals * lambda_vals_sub, x=k_r_m, axis=-1)
    #     vertical_bf = np.nanmean(vertical_bf) # average in depth
    # else:
    #     vertical_bf = np.nan
    # vertical_bf_submeso_mld.append(vertical_bf)

    if not np.isnan(Hml_this_week):
        # spec_vals: shape (Z, kr)
        spec_vals = spec_submeso_mld.fillna(0.0).values  # (Z, kr)
        # Make sure k_r_submeso has shape (kr,) and broadcast it to (1, kr)
        k_r_broadcast = k_r_submeso[None, :]  # shape (1, kr)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cospec = spec_vals / k_r_broadcast  # shape (Z, kr)
        # Integrate over kr (axis=-1), result shape: (Z,)
        vertical_bf_profile = np.trapezoid(cospec, x=k_r_submeso , axis=-1)

        # Vertical average over MLD (shape Z,) → scalar
        # vertical_bf = np.nanmean(vertical_bf_profile)
        depth_vals_meanVBF = spec_submeso_mld['Z'].values 
        vertical_bf_integral = np.trapezoid(vertical_bf_profile, x=depth_vals_meanVBF)
        vertical_bf = vertical_bf_integral / np.trapezoid(np.ones_like(depth_vals_meanVBF), x=depth_vals_meanVBF)

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

    # # # ==========================
    # # --- Plot ---
    # # ==========================
    # fig, ax = plt.subplots(figsize=(8, 6))
    # norm = TwoSlopeNorm(vcenter=1e-10, vmin=0, vmax=5e-10)
    # pc = ax.pcolormesh(
    #     k_r_filtered, -spec_filtered.Z, spec_filtered,
    #     shading='auto', cmap=cmap, norm=norm
    # )
    # ax.invert_yaxis()
    # ax.set_xscale('log')
    # ax.set_title(f'$w$-$b$ Cross-Spectrum (VP), {date_str}')
    # ax.set_xlabel('Wavenumber $k_r$ (cpkm)')
    # ax.set_ylabel('Depth (m)')

    # # 20 km cutoff line at bottom
    # ax.axvline(kr_cutoff_meso, color='g', linestyle='--')
    # ax.text(kr_cutoff_meso, 0.05, '20 km cutoff', color='g', ha='right',
    #         va='bottom', rotation=90, transform=ax.get_xaxis_transform())

    # fig.colorbar(pc, ax=ax, label=r'Spectral density (ms$^{-3}$)')
    # plt.tight_layout()
    # plt.savefig(f"{figdir}/wb_cross-spectrum_{date_str}.png", dpi=150)
    # plt.close()







# # ========== Convert PNGs to MP4 ==========

# mp4_path = os.path.join(figdir, "movie-wb_spectra_7d_rolling.mp4")
# os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/wb_cross-spectrum_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {mp4_path}")
# print(f"🎬 Saved movie to {mp4_path}")



target_length = max(arr.shape[0] for arr in mean_spec_mld_by_kr if isinstance(arr, np.ndarray))
mean_spec_mld_by_kr_padded = [
    np.pad(arr, (0, target_length - arr.shape[0]), constant_values=np.nan)
    if arr.shape[0] < target_length else arr
    for arr in mean_spec_mld_by_kr
]
mean_spec_mld_by_kr_np = np.array(mean_spec_mld_by_kr_padded)  # shape: (time, kr)

kr_peak_np = np.array(kr_peak_list)

# ========== Plot MLD-averaged spectrum vs kr ==========
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(0, len(times), max(1, len(times)//20)):  # Plot subset for clarity
    ax.plot(k_r, mean_spec_mld_by_kr_np[i], label=str(np.datetime_as_string(times[i], unit='D')))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Wavenumber $k_r$ (cpkm)")
ax.set_ylabel("Mean Spectrum in MLD (m²/s³)")
ax.set_title("MLD-Averaged $w$-$b$ Spectrum vs Wavenumber")
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.grid(True, which='both')
plt.tight_layout()
plt.savefig(f"{figdir}/mean_spec_in_MLD_vs_kr.png", dpi=150)
plt.close()


# ========== Plot kr_peak vs time ==========
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, kr_peak_np, marker='o', color='tab:red')
ax.set_title('Peak Wavenumber of MLD-Averaged Spectrum')
ax.set_ylabel('Peak $k_r$ (cpkm)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/kr_peak_timeseries.png", dpi=150)
plt.close()


import matplotlib.dates as mdates

# Calculate wavelength from peak wavenumber (avoid division by zero)
wavelength = 1 / kr_peak_np
# Optional: handle zeros or negative values in kr_peak_np if any

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, wavelength, marker='o', color='tab:blue')  # changed color for clarity
ax.set_title('Peak Wavelength of MLD-Averaged Spectrum')
ax.set_ylabel('Peak Wavelength (units)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/wavelength_peak_timeseries.png", dpi=150)
plt.close()









import numpy as np
import pandas as pd
import matplotlib.dates as mdates

# Calculate wavelength from peak wavenumber
wavelength = 1 / kr_peak_np
# Create a DataFrame for easier filtering and smoothing
df = pd.DataFrame({'time': times, 'wavelength': wavelength})
df.set_index('time', inplace=True)
# Remove wavelengths > 50 km
df = df[df['wavelength'] <= 50]
# Apply a 7-day rolling mean (assuming 'times' is a datetime index)
# If times are irregular, this will do a window of 7 time points; for exact days, you can use time-based windows if times are datetime
df_smooth = df.rolling('7D').mean()
# Plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_smooth.index, df_smooth['wavelength'], marker='o', color='tab:blue')
ax.set_title('Smoothed Peak Wavelength (7-day Rolling Mean)')
ax.set_ylabel('Peak Wavelength (km)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/wavelength_peak_timeseries_smoothed.png", dpi=150)
plt.close()











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




import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Create the original DataFrame
df = pd.DataFrame({'time': time_pd, 'wavelength': energy_weighted_Lr})
df.set_index('time', inplace=True)

# Keep only valid wavelength values for interpolation preparation
df_valid = df[(df['wavelength'] >= 0) & (df['wavelength'] <= 50)]

# Reindex to ensure all original time points are included (even the ones removed during filtering)
df_interp = df.reindex(df.index)

# Set invalid values to NaN (so they can be interpolated)
df_interp.loc[(df_interp['wavelength'] < 0) | (df_interp['wavelength'] > 50), 'wavelength'] = pd.NA

# Interpolate missing values based on time (linear interpolation)
df_interp['wavelength'] = df_interp['wavelength'].interpolate(method='time')

# Apply 7-day rolling mean smoothing
df_interp['wavelength_smoothed'] = df_interp['wavelength'].rolling(window=7, center=True, min_periods=1).mean()

# Plot the smoothed, interpolated data
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_interp.index, df_interp['wavelength_smoothed'], marker='o', color='tab:green')
ax.set_title('Variance-Weighted Mean Wavelength ($L_r$) - 7-day Smoothed (with Interpolation)')
ax.set_ylabel('Wavelength (km)')
ax.set_xlabel('Time')
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(f"{figdir}/energy_weighted_Lr_timeseries_interpolated_smoothed.png", dpi=150)
plt.close()

print("📏 Interpolated and smoothed energy-weighted wavelength plot saved.")








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






# ==========================
# Save all results to NetCDF (final version)
# ==========================

# Convert lists to arrays
time_np = np.array(times)

# Prepare smoothed DataFrames as arrays
# 1. Energy-weighted Lr smoothed
energy_weighted_Lr_smoothed = df_interp['wavelength_smoothed'].reindex(time_pd).values

# 2. Wavelength peak smoothed
wavelength_peak_smoothed = df_smooth['wavelength'].reindex(time_pd).values

# Create xarray Dataset
ds_out = xr.Dataset(
    data_vars=dict(
        mean_spec_in_mld=(["time"], mean_spec_in_mld),
        mean_spec_in_mld_submeso=(["time"], mean_spec_in_mld_submeso),
        min_spec_val_in_mld=(["time"], max_values),
        depth_at_min_spec=(["time"], max_depths),
        kr_at_min_spec=(["time"], max_krs),
        Lr_at_min_spec=(["time"], max_Lr),
        vertical_bf_submeso_mld=(["time"], vertical_bf_submeso_mld),
        energy_weighted_Lr=(["time"], energy_weighted_Lr),
        energy_weighted_Lr_smoothed=(["time"], energy_weighted_Lr_smoothed),
        kr_peak=(["time"], kr_peak_np),
        wavelength_peak=(["time"], 1 / kr_peak_np),
        wavelength_peak_smoothed=(["time"], wavelength_peak_smoothed),
        mean_spec_mld_by_kr=(["time", "k_r"], mean_spec_mld_by_kr_np)
    ),
    coords=dict(
        time=("time", time_pd),
        k_r=("k_r", k_r.data)
    ),
    attrs=dict(
        description="Derived metrics from w-b cross-spectrum analysis (7-day rolling mean)",
        kr_cutoff=f"{kr_cutoff} cpkm (>500 km excluded)",
        kr_cutoff_meso=f"{kr_cutoff_meso} cpkm (<20 km = submesoscale)"
    )
)

# Save to NetCDF
ds_out.to_netcdf(out_nc_path)
print(f"💾 Final output with smoothed time series saved to {out_nc_path}")




print("✅ All plots completed.")

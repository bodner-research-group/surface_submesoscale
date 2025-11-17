##### Run this script on an interactive node
##### Compute gradient of sea surface height anomaly, using 24-h averaged data

import xarray as xr
import numpy as np
import os
from xgcm import Grid
import gsw

from set_constant import domain_name, face, i, j

from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)


# ========= Paths =========
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
TS_24h_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"
output_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surfaceTS_gradient"
os.makedirs(output_path, exist_ok=True)

# ========= Load grid data =========
# print("Loading grid...")
ds1 = xr.open_zarr(grid_path, consolidated=False)
ds_grid_face = ds1.isel(face=face,i=i, j=j,i_g=i, j_g=j,k=0,k_p1=0,k_u=0)

# Drop time dimension if exists
if 'time' in ds_grid_face.dims:
    ds_grid_face = ds_grid_face.isel(time=0, drop=True)  # or .squeeze('time')

# ========= Setup xgcm grid =========
coords = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
}
metrics = {
    ("X",): ["dxC", "dxG"],
    ("Y",): ["dyC", "dyG"],
}
grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)

lon = ds1.XC.isel(face=face,i=i, j=j)
lat = ds1.YC.isel(face=face,i=i, j=j)


# ========= Load daily averaged tt_s =========
print("Loading daily averaged tt_s and ss_s...")

tt_s_path = os.path.join(TS_24h_dir, "tt_24h_*.nc")
ds_tt_s = xr.open_mfdataset(tt_s_path, combine='by_coords')
tt_s = ds_tt_s["Theta"].isel(k=0) # Align datasets and select face/i/j region

ss_s_path = os.path.join(TS_24h_dir, "ss_24h_*.nc")
ds_ss_s = xr.open_mfdataset(ss_s_path, combine='by_coords')
ss_s = ds_ss_s["Salt"].isel(k=0) # Align datasets and select face/i/j region

# ========= Compute surface density =========
SA_s = gsw.SA_from_SP(ss_s, 0, lon, lat)
CT_s = gsw.CT_from_pt(SA_s, tt_s)
rho_s = gsw.rho(SA_s, CT_s, 0)

alpha_s = gsw.alpha(SA_s, CT_s, 0) ## alpha_s.mean().values = 1.601e-4
beta_s = gsw.beta(SA_s, CT_s, 0)   ## beta_s.mean().values = 7.552e-4

tAlpha_ref = 2.0e-4
sBeta_ref = 7.4e-4

gravity = 9.81
rho0 = 1027.5
T0_const = 20
S0_const = 30

buoy_s = -gravity*(rho_s-rho0)/rho0
# buoy_s_linear = gravity*alpha_s*(CT_s-T0_const) - gravity*beta_s*(SA_s-S0_const)
buoy_s_linear = gravity*tAlpha_ref*(CT_s-T0_const) - gravity*sBeta_ref*(SA_s-S0_const)


# ========= Compute derivatives =========
tt_s_x = grid.derivative(tt_s, axis="X") # ∂tt_s/∂x
tt_s_y = grid.derivative(tt_s, axis="Y") # ∂tt_s/∂y

ss_s_x = grid.derivative(ss_s, axis="X") # ∂ss_s/∂x
ss_s_y = grid.derivative(ss_s, axis="Y") # ∂ss_s/∂y

rho_s_x = grid.derivative(rho_s, axis="X") # ∂rho_s/∂x
rho_s_y = grid.derivative(rho_s, axis="Y") # ∂rho_s/∂y

buoy_s_x = grid.derivative(buoy_s, axis="X") # ∂buoy_s/∂x
buoy_s_y = grid.derivative(buoy_s, axis="Y") # ∂buoy_s/∂y

buoy_s_linear_x = grid.derivative(buoy_s_linear, axis="X") # ∂rho_s_linear/∂x
buoy_s_linear_y = grid.derivative(buoy_s_linear, axis="Y") # ∂rho_s_linear/∂y

# SST and SSS gradient magnitude at center
tt_s_x_center = grid.interp(tt_s_x, axis="X", to="center")
tt_s_y_center = grid.interp(tt_s_y, axis="Y", to="center")
tt_s_grad_mag = np.sqrt(tt_s_x_center**2 + tt_s_y_center**2)
tt_s_grad_mag = tt_s_grad_mag.assign_coords(time=tt_s.time)

ss_s_x_center = grid.interp(ss_s_x, axis="X", to="center")
ss_s_y_center = grid.interp(ss_s_y, axis="Y", to="center")
ss_s_grad_mag = np.sqrt(ss_s_x_center**2 + ss_s_y_center**2)
ss_s_grad_mag = ss_s_grad_mag.assign_coords(time=ss_s.time)

rho_s_x_center = grid.interp(rho_s_x, axis="X", to="center")
rho_s_y_center = grid.interp(rho_s_y, axis="Y", to="center")
rho_s_grad_mag = np.sqrt(rho_s_x_center**2 + rho_s_y_center**2)
rho_s_grad_mag = rho_s_grad_mag.assign_coords(time=rho_s.time)

buoy_s_x_center = grid.interp(buoy_s_x, axis="X", to="center")
buoy_s_y_center = grid.interp(buoy_s_y, axis="Y", to="center")
buoy_s_grad_mag = np.sqrt(buoy_s_x_center**2 + buoy_s_y_center**2)
buoy_s_grad_mag = buoy_s_grad_mag.assign_coords(time=buoy_s.time)

buoy_s_linear_x_center = grid.interp(buoy_s_linear_x, axis="X", to="center")
buoy_s_linear_y_center = grid.interp(buoy_s_linear_y, axis="Y", to="center")
buoy_s_linear_grad_mag = np.sqrt(buoy_s_linear_x_center**2 + buoy_s_linear_y_center**2)
buoy_s_linear_grad_mag = buoy_s_linear_grad_mag.assign_coords(time=buoy_s_linear.time)


# Compute the mean tt_s_grad_mag and ss_s_grad_mag over X and Y
# tt_s_grad_mag_daily = tt_s_grad_mag.mean(dim=["i","j"])
# tt_s_grad_mag_weekly = tt_s_grad_mag_daily.rolling(time=7, center=True).mean()

# ss_s_grad_mag_daily = ss_s_grad_mag.mean(dim=["i","j"])
# ss_s_grad_mag_weekly = ss_s_grad_mag_daily.rolling(time=7, center=True).mean()

from dask.diagnostics import ProgressBar

with ProgressBar():
    tt_s_grad_mag_daily = tt_s_grad_mag.mean(dim=["i", "j"]).compute()
    tt_s_grad_mag_weekly = tt_s_grad_mag_daily.rolling(time=7, center=True).mean()

    ss_s_grad_mag_daily = ss_s_grad_mag.mean(dim=["i", "j"]).compute()
    ss_s_grad_mag_weekly = ss_s_grad_mag_daily.rolling(time=7, center=True).mean()

    rho_s_grad_mag_daily = rho_s_grad_mag.mean(dim=["i", "j"]).compute()
    rho_s_grad_mag_weekly = rho_s_grad_mag_daily.rolling(time=7, center=True).mean()

    buoy_s_grad_mag_daily = buoy_s_grad_mag.mean(dim=["i", "j"]).compute()
    buoy_s_grad_mag_weekly = buoy_s_grad_mag_daily.rolling(time=7, center=True).mean()

    buoy_s_linear_grad_mag_daily = buoy_s_linear_grad_mag.mean(dim=["i", "j"]).compute()
    buoy_s_linear_grad_mag_weekly = buoy_s_linear_grad_mag_daily.rolling(time=7, center=True).mean()

    alpha_s_daily = alpha_s.mean(dim=["i", "j"]).compute()
    alpha_s_weekly = alpha_s_daily.rolling(time=7, center=True).mean()

    beta_s_daily = beta_s.mean(dim=["i", "j"]).compute()
    beta_s_weekly = beta_s_daily.rolling(time=7, center=True).mean()


# ========= Save results =========
ds_out = xr.Dataset({
    "tt_s_grad_mag_daily": tt_s_grad_mag_daily,
    "tt_s_grad_mag_weekly": tt_s_grad_mag_weekly,
    "ss_s_grad_mag_daily": ss_s_grad_mag_daily,
    "ss_s_grad_mag_weekly": ss_s_grad_mag_weekly,
    "rho_s_grad_mag_daily": rho_s_grad_mag_daily,
    "rho_s_grad_mag_weekly": rho_s_grad_mag_weekly,
    "buoy_s_grad_mag_daily": buoy_s_grad_mag_daily,
    "buoy_s_grad_mag_weekly": buoy_s_grad_mag_weekly,
    "buoy_s_linear_grad_mag_daily": buoy_s_linear_grad_mag_daily,
    "buoy_s_linear_grad_mag_weekly": buoy_s_linear_grad_mag_weekly,
    "alpha_s_daily": alpha_s_daily,
    "alpha_s_weekly": alpha_s_weekly,
    "beta_s_daily": beta_s_daily,
    "beta_s_weekly": beta_s_weekly,
})

output_file = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}", "surfaceT-S-rho-buoy-alpha-beta_gradient_magnitude.nc")
# ds_out.to_netcdf(output_file)

encoding = {
    var: {"zlib": True, "complevel": 4, "chunksizes": (100,)} 
    for var in ds_out.data_vars
}

with ProgressBar():
    ds_out.to_netcdf(output_file, encoding=encoding)


print(f"Done: daily/weekly SST/SSS/surface density gradient magnitude saved to {output_file}")





#################################
########### Make plots ##########
#################################
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from set_colormaps import WhiteBlueGreenYellowRed
cmap = WhiteBlueGreenYellowRed()

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}"


# gravity = 9.81
# rho0 = 1027.5
# buoy_s_grad_mag_daily = -gravity*rho_s_grad_mag_daily/rho0
# buoy_s_grad_mag_weekly = -gravity*rho_s_grad_mag_weekly/rho0

buoy_s_grad_mag_AnnualMean = float(buoy_s_grad_mag_weekly.mean())

vmax = buoy_s_grad_mag_AnnualMean*4
threshold = buoy_s_grad_mag_AnnualMean*2

# buoy_s_grad_mag = -gravity*rho_s_grad_mag/rho0

# Calculate the ratio of area with SST gradient magnitude higher than the threshold
above_threshold_ratio = (buoy_s_grad_mag > threshold).mean(dim=["i", "j"])

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)


buoy_s_grad_mag_daily.plot(ax=axs[0], label='Daily Mean')
buoy_s_grad_mag_weekly.plot(ax=axs[0], label='Weekly Rolling Mean', linewidth=2)
axs[0].set_title("Surface Buoyancy Gradient Magnitude")
axs[0].set_ylabel("|∇b| (1/s^2)")
axs[0].legend()
axs[0].grid(True)

above_threshold_ratio.plot(ax=axs[1], color='tab:green', linewidth=2)
axs[1].set_title("Fraction of Area with High surface |∇b|")
axs[1].set_ylabel("Fraction")
axs[1].set_xlabel("Time")
axs[1].grid(True)

### Add labels and minor grid lines
axs[0].text(0.01, 0.95, "(a)", transform=axs[0].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')
axs[1].text(0.01, 0.95, "(b)", transform=axs[1].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')

axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(f"{figdir}/surface_buoyancy_gradient_timeseries.png", dpi=200)
plt.close()






tt_s_grad_mag_AnnualMean = float(tt_s_grad_mag_weekly.mean())

vmax = tt_s_grad_mag_AnnualMean*4
threshold = tt_s_grad_mag_AnnualMean*2

# Calculate the ratio of area with SST gradient magnitude higher than the threshold
above_threshold_ratio = (tt_s_grad_mag > threshold).mean(dim=["i", "j"])

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

tt_s_grad_mag_daily.plot(ax=axs[0], label='Daily Mean')
tt_s_grad_mag_weekly.plot(ax=axs[0], label='Weekly Rolling Mean', linewidth=2)
axs[0].set_title("SST Gradient Magnitude: Daily & Weekly Mean")
axs[0].set_ylabel("|∇SST| (m/m)")
axs[0].legend()
axs[0].grid(True)

above_threshold_ratio.plot(ax=axs[1], color='tab:green', linewidth=2)
axs[1].set_title("Fraction of Area with High SST Gradient")
axs[1].set_ylabel("Fraction")
axs[1].set_xlabel("Time")
axs[1].grid(True)

### Add labels and minor grid lines
axs[0].text(0.01, 0.95, "(a)", transform=axs[0].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')
axs[1].text(0.01, 0.95, "(b)", transform=axs[1].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')

axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(f"{figdir}/SST_gradient_timeseries.png", dpi=200)
plt.close()






ss_s_grad_mag_AnnualMean = float(ss_s_grad_mag_weekly.mean())

vmax = ss_s_grad_mag_AnnualMean*4
threshold = ss_s_grad_mag_AnnualMean*2

# Calculate the ratio of area with SST gradient magnitude higher than the threshold
above_threshold_ratio = (ss_s_grad_mag > threshold).mean(dim=["i", "j"])

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ss_s_grad_mag_daily.plot(ax=axs[0], label='Daily Mean')
ss_s_grad_mag_weekly.plot(ax=axs[0], label='Weekly Rolling Mean', linewidth=2)
axs[0].set_title("SSS Gradient Magnitude: Daily & Weekly Mean")
axs[0].set_ylabel("|∇SSS| (m/m)")
axs[0].legend()
axs[0].grid(True)

above_threshold_ratio.plot(ax=axs[1], color='tab:green', linewidth=2)
axs[1].set_title("Fraction of Area with High SSS Gradient")
axs[1].set_ylabel("Fraction")
axs[1].set_xlabel("Time")
axs[1].grid(True)

### Add labels and minor grid lines
axs[0].text(0.01, 0.95, "(a)", transform=axs[0].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')
axs[1].text(0.01, 0.95, "(b)", transform=axs[1].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')

axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(f"{figdir}/SSS_gradient_timeseries.png", dpi=200)
plt.close()







# # ========================================
# # Compute Non-Overlapping Weekly PDFs
# # ========================================
# weekly_pdf_dir = f"{figdir}/SST_gradient_weekly_PDFs"
# os.makedirs(weekly_pdf_dir, exist_ok=True)

# pdf_data_dir = f"{output_path}/weekly_PDF_data_SST"
# os.makedirs(pdf_data_dir, exist_ok=True)

# n_days = tt_s_grad_mag.sizes['time']
# week_length = 7
# n_weeks = n_days // week_length

# for week_idx in range(n_weeks):
#     start = week_idx * week_length
#     end = start + week_length

#     tt_s_week = tt_s_grad_mag.isel(time=slice(start, end))
#     date_str = str(tt_s_week.time[0].values)[:10]  # First day of the week

#     # Flatten all values from 7 days and spatial dims
#     grad_flat = tt_s_week.values.reshape(-1)
#     grad_flat = grad_flat[~np.isnan(grad_flat)]

#     if grad_flat.size == 0:
#         print(f"Week {date_str} has no valid data. Skipping.")
#         continue

#     # Define log-spaced bins
#     min_val = grad_flat.min()
#     max_val = grad_flat.max()
#     bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    
#     # Compute PDF
#     pdf_vals, bin_edges = np.histogram(grad_flat, bins=bins, density=True)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     # # Save data
#     # np.savez(os.path.join(pdf_data_dir, f"PDF_week_{date_str}.npz"),
#     #          bin_centers=bin_centers,
#     #          pdf_vals=pdf_vals)
    
#     pdf_vals = pdf_vals+1e-10;

#     # # Plot linear PDF
#     # plt.figure(figsize=(8, 6))
#     # plt.plot(bin_centers, pdf_vals, color='black', linewidth=2)
#     # plt.title(f"PDF of SST Gradient Magnitude\nWeek Starting: {date_str}")
#     # plt.xlabel("|∇SST| (m/m)")
#     # plt.ylabel("Probability Density")
#     # plt.grid(True)
#     # plt.xscale('linear')  # Linear x-axis
#     # plt.yscale('log')
#     # plt.xlim(1e-8, 1e-2)
#     # plt.ylim(0.1, 1e5)
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(weekly_pdf_dir, f"SST_grad_PDF_week_{date_str}.png"), dpi=150)
#     # plt.close()

#     # Plot log-log PDF
#     plt.figure(figsize=(8, 6))
#     plt.plot(bin_centers, pdf_vals, color='black', linewidth=2)
#     plt.title(f"PDF of SST Gradient Magnitude (Log-Log)\nWeek Starting: {date_str}")
#     plt.xlabel("|∇SST| (m/m)")
#     plt.ylabel("Probability Density")
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlim(1e-8, 1e-2)
#     plt.ylim(0.1, 1e5)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(weekly_pdf_dir, f"SST_grad_PDF_loglog_week_{date_str}.png"), dpi=150)
#     plt.close()

#     print(f"Saved log-binned weekly PDF and plots for week starting {date_str}")






# # ========================================
# # Compute Non-Overlapping Weekly PDFs
# # ========================================
# weekly_pdf_dir = f"{figdir}/SSS_gradient_weekly_PDFs"
# os.makedirs(weekly_pdf_dir, exist_ok=True)

# pdf_data_dir = f"{output_path}/weekly_PDF_data_SSS"
# os.makedirs(pdf_data_dir, exist_ok=True)

# n_days = tt_s_grad_mag.sizes['time']
# week_length = 7
# n_weeks = n_days // week_length

# for week_idx in range(n_weeks):
#     start = week_idx * week_length
#     end = start + week_length

#     tt_s_week = tt_s_grad_mag.isel(time=slice(start, end))
#     date_str = str(tt_s_week.time[0].values)[:10]  # First day of the week

#     # Flatten all values from 7 days and spatial dims
#     grad_flat = tt_s_week.values.reshape(-1)
#     grad_flat = grad_flat[~np.isnan(grad_flat)]

#     if grad_flat.size == 0:
#         print(f"Week {date_str} has no valid data. Skipping.")
#         continue

#     # Define log-spaced bins
#     min_val = grad_flat.min()
#     max_val = grad_flat.max()
#     bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    
#     # Compute PDF
#     pdf_vals, bin_edges = np.histogram(grad_flat, bins=bins, density=True)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     # # Save data
#     # np.savez(os.path.join(pdf_data_dir, f"PDF_week_{date_str}.npz"),
#     #          bin_centers=bin_centers,
#     #          pdf_vals=pdf_vals)
    
#     pdf_vals = pdf_vals+1e-10; ### avoid log(0)

#     # # Plot linear PDF
#     # plt.figure(figsize=(8, 6))
#     # plt.plot(bin_centers, pdf_vals, color='black', linewidth=2)
#     # plt.title(f"PDF of SSS Gradient Magnitude\nWeek Starting: {date_str}")
#     # plt.xlabel("|∇SSS| (m/m)")
#     # plt.ylabel("Probability Density")
#     # plt.grid(True)
#     # plt.xscale('linear')  # Linear x-axis
#     # plt.yscale('log')
#     # plt.xlim(1e-8, 1e-2)
#     # plt.ylim(0.1, 1e5)
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(weekly_pdf_dir, f"SSS_grad_PDF_week_{date_str}.png"), dpi=150)
#     # plt.close()

#     # Plot log-log PDF
#     plt.figure(figsize=(8, 6))
#     plt.plot(bin_centers, pdf_vals, color='black', linewidth=2)
#     plt.title(f"PDF of SSS Gradient Magnitude (Log-Log)\nWeek Starting: {date_str}")
#     plt.xlabel("|∇SSS| (m/m)")
#     plt.ylabel("Probability Density")
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlim(1e-8, 1e-2)
#     plt.ylim(0.1, 1e5)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(weekly_pdf_dir, f"SSS_grad_PDF_loglog_week_{date_str}.png"), dpi=150)
#     plt.close()

#     print(f"Saved log-binned weekly PDF and plots for week starting {date_str}")



# ##### Convert images to video
# import os
# figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SST_gradient_weekly_PDFs"
# # high-resolution
# output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SST_grad_PDF_loglog.mp4"
# os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/SST_grad_PDF_loglog_week_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")



# ##### Convert images to video
# import os
# figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSS_gradient_weekly_PDFs"
# # high-resolution
# output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SSS_grad_PDF_loglog.mp4"
# os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/SSS_grad_PDF_loglog_week_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")





# # ================================
# # Weekly Histograms of SST and SSS Gradients
# # ================================
# weekly_hist_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/weekly_gradient_histograms"
# os.makedirs(weekly_hist_dir, exist_ok=True)

# print("Generating weekly histograms of SST and SSS gradients...")

# tt_grad_min = float(tt_s_grad_mag.min().values)
# tt_grad_max = float(tt_s_grad_mag.max().values)

# ss_grad_min = float(ss_s_grad_mag.min().values)
# ss_grad_max = float(ss_s_grad_mag.max().values)

# print(f"Global SST gradient magnitude range: {tt_grad_min:.2e} to {tt_grad_max:.2e}")
# print(f"Global SSS gradient magnitude range: {ss_grad_min:.2e} to {ss_grad_max:.2e}")

# for week_idx in range(n_weeks):
#     start = week_idx * week_length
#     end = start + week_length

#     tt_grad_week = tt_s_grad_mag.isel(time=slice(start, end));
#     ss_grad_week = ss_s_grad_mag.isel(time=slice(start, end));

#     date_str = str(tt_grad_week.time[0].values)[:10];

#     # flatten and remove NaNs
#     tt_grad_flat = tt_grad_week.values.reshape(-1);
#     ss_grad_flat = ss_grad_week.values.reshape(-1);

#     tt_grad_flat = tt_grad_flat[~np.isnan(tt_grad_flat)];
#     ss_grad_flat = ss_grad_flat[~np.isnan(ss_grad_flat)];

#     # Add small value to avoid zeros in bins
#     epsilon = 1e-2
#     tt_hist, tt_bins = np.histogram(tt_grad_flat, bins=100)
#     tt_hist = tt_hist + epsilon  # Avoid log(0)

#     plt.figure(figsize=(8, 6))
#     plt.bar(tt_bins[:-1], tt_hist, width=np.diff(tt_bins), color='royalblue', alpha=0.8, edgecolor='black', log=True)
#     plt.title(f"Histogram of SST Gradient Magnitude\nWeek Starting: {date_str}")
#     plt.xlabel("|∇SST| (m/m)")
#     plt.ylabel("Frequency (log scale)")
#     plt.xlim(tt_grad_min, tt_grad_max)
#     plt.ylim(0.5, 1e6)  # Should now work
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(weekly_hist_dir, f"SST_grad_hist_week_{date_str}.png"), dpi=150)
#     plt.close()


#     # Add small value to avoid zeros in bins
#     epsilon = 1e-2
#     ss_hist, ss_bins = np.histogram(ss_grad_flat, bins=100)
#     ss_hist = ss_hist + epsilon  # Avoid log(0)

#     plt.figure(figsize=(8, 6))
#     plt.bar(ss_bins[:-1], ss_hist, width=np.diff(ss_bins), color='royalblue', alpha=0.8, edgecolor='black', log=True)
#     plt.title(f"Histogram of SSS Gradient Magnitude\nWeek Starting: {date_str}")
#     plt.xlabel("|∇SSS| (m/m)")
#     plt.ylabel("Frequency (log scale)")
#     plt.xlim(ss_grad_min, ss_grad_max)
#     plt.ylim(0.5, 1e6)  # Should now work
#     plt.xticks(rotation=45)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(weekly_hist_dir, f"SSS_grad_hist_week_{date_str}.png"), dpi=150)
#     plt.close()


#     print(f"Saved weekly SST and SSS gradient magnitude histograms for week starting {date_str}")




# ##### Convert images to video
# import os
# figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/weekly_gradient_histograms"
# # high-resolution
# output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SST_grad_hist.mp4"
# os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/SST_grad_hist_week_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")



# ##### Convert images to video
# import os
# figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/weekly_gradient_histograms"
# # high-resolution
# output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SSS_grad_hist.mp4"
# os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/SSS_grad_hist_week_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")



# # # ================================
# # # Weekly Histograms of SST and SSS
# # # ================================
# # weekly_hist_dir = f"{figdir}/weekly_histograms"
# # os.makedirs(weekly_hist_dir, exist_ok=True)

# # print("Generating weekly SST and SSS histograms...")

# # tt_min = float(tt_s.min().values)
# # tt_max = float(tt_s.max().values)

# # ss_min = float(ss_s.min().values)
# # ss_max = float(ss_s.max().values)

# # print(f"Global SST range: {tt_min:.2f} to {tt_max:.2f}")
# # print(f"Global SSS range: {ss_min:.2f} to {ss_max:.2f}")


# # for week_idx in range(n_weeks):
# #     start = week_idx * week_length;
# #     end = start + week_length;

# #     tt_week = tt_s.isel(time=slice(start, end));
# #     ss_week = ss_s.isel(time=slice(start, end));

# #     date_str = str(tt_week.time[0].values)[:10];

# #     # Flatten and clean data
# #     tt_flat = tt_week.values.reshape(-1);
# #     ss_flat = ss_week.values.reshape(-1);

# #     tt_flat = tt_flat[~np.isnan(tt_flat)];
# #     ss_flat = ss_flat[~np.isnan(ss_flat)];
    

# #     if tt_flat.size == 0 or ss_flat.size == 0:
# #         print(f"Week {date_str} has no valid SST or SSS data. Skipping.")
# #         continue

# #     # Plot histogram for SST
# #     plt.figure(figsize=(8, 6))
# #     plt.hist(tt_flat, bins=100, color='royalblue', alpha=0.8, edgecolor='black')
# #     plt.title(f"Histogram of SST\nWeek Starting: {date_str}")
# #     plt.xlabel("SST (°C)")
# #     plt.ylabel("Frequency")
# #     plt.xlim(tt_min, tt_max) 
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(weekly_hist_dir, f"SST_hist_week_{date_str}.png"), dpi=150)
# #     plt.close()

# #     # Plot histogram for SSS
# #     plt.figure(figsize=(8, 6))
# #     plt.hist(ss_flat, bins=100, color='darkorange', alpha=0.8, edgecolor='black')
# #     plt.title(f"Histogram of SSS\nWeek Starting: {date_str}")
# #     plt.xlabel("SSS (psu)")
# #     plt.ylabel("Frequency")
# #     plt.grid(True)
# #     plt.xlim(ss_min, ss_max)
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(weekly_hist_dir, f"SSS_hist_week_{date_str}.png"), dpi=150)
# #     plt.close()

# #     print(f"Saved weekly SST and SSS histograms for week starting {date_str}")
##### Run this script on an interactive node
##### Compute gradient of sea surface height anomaly, using 24-h averaged data

import xarray as xr
import numpy as np
import os
from xgcm import Grid

from set_constant import domain_name, face, i, j

# ========= Paths =========
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
output_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_gradient"
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


# ========= Load daily averaged Eta =========
# print("Loading daily averaged Eta...")
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")

ds_eta = xr.open_mfdataset(eta_path, combine='by_coords')

# Align datasets and select face/i/j region
Eta = ds_eta["Eta"]

# ========= Compute derivatives =========
# ∂Eta/∂x
eta_x = grid.derivative(Eta, axis="X")

# ∂Eta/∂y
eta_y = grid.derivative(Eta, axis="Y")

# SSH gradient magnitude at center
eta_x_center = grid.interp(eta_x, axis="X", to="center")
eta_y_center = grid.interp(eta_y, axis="Y", to="center")

eta_grad_mag = np.sqrt(eta_x_center**2 + eta_y_center**2)

eta_grad_mag = eta_grad_mag.assign_coords(time=Eta.time)

# Compute the mean eta_grad_mag over X and Y
eta_grad_mag_daily = eta_grad_mag.mean(dim=["i","j"])
eta_grad_mag_weekly = eta_grad_mag_daily.rolling(time=7, center=True).mean()

# ========= Save results =========
ds_out = xr.Dataset({
    "eta_grad_mag_daily": eta_grad_mag_daily,
    "eta_grad_mag_weekly": eta_grad_mag_weekly,
})

output_file = os.path.join(output_path, "SSH_gradient_magnitude.nc")
ds_out.to_netcdf(output_file)

print(f"Done: daily/weekly SSH gradient magnitude saved to {output_file}")






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


eta_grad_mag_AnnualMean = float(eta_grad_mag_weekly.mean())

vmax = eta_grad_mag_AnnualMean*4
threshold = eta_grad_mag_AnnualMean*2

# Calculate the ratio of area with SSH gradient magnitude higher than the threshold
above_threshold_ratio = (eta_grad_mag > threshold).mean(dim=["i", "j"])


fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

eta_grad_mag_daily.plot(ax=axs[0], label='Daily Mean')
eta_grad_mag_weekly.plot(ax=axs[0], label='Weekly Rolling Mean', linewidth=2)
axs[0].set_title("SSH Gradient Magnitude: Daily & Weekly Mean")
axs[0].set_ylabel("|∇η| (m/m)")
axs[0].legend()
axs[0].grid(True)

above_threshold_ratio.plot(ax=axs[1], color='tab:green', linewidth=2)
axs[1].set_title("Fraction of Area with High SSH Gradient")
axs[1].set_ylabel("Fraction")
axs[1].set_xlabel("Time")
axs[1].grid(True)

### Add labels and minor grid lines
axs[0].text(0.01, 0.95, "(a)", transform=axs[0].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')
axs[1].text(0.01, 0.95, "(b)", transform=axs[1].transAxes, fontsize=13,
            verticalalignment='top', fontweight='bold')

# axs[0].grid(True, which='major', linestyle='-', linewidth=0.5)
# axs[0].grid(True, which='minor', linestyle=':', linewidth=0.3)
# axs[0].xaxis.set_minor_locator(mdates.DayLocator(interval=7))

# axs[1].grid(True, which='major', linestyle='-', linewidth=0.5)
# axs[1].grid(True, which='minor', linestyle=':', linewidth=0.3)
# axs[1].xaxis.set_minor_locator(mdates.DayLocator(interval=7))

axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(f"{figdir}/SSH_gradient_timeseries.png", dpi=200)
plt.close()


###### Plot SSH gradient of each week and make an animation

# lon = ds_grid_face['XC']
# lat = ds_grid_face['YC']

lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
lat_vals = lat.values  # shape (j,)
lon_vals = lon.values  # shape (i,)
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')  # shape (j, i)


### Use Dask
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import matplotlib
import os


cluster = LocalCluster(n_workers=20, threads_per_worker=1, memory_limit="18GB")
# cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
# print("Dask dashboard:", client.dashboard_link)

# Plot SSH gradient magnitude 
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_gradient"
os.makedirs(figdir, exist_ok=True)

# for t in range(len(eta_grad_mag["time"])):
#     eta_grad_2d = eta_grad_mag.isel(time=t)
#     date_str = str(eta_grad_mag.time[t].values)[:10]

#     fig = plt.figure(figsize=(6, 5))
#     im = plt.pcolormesh(lon, lat, eta_grad_2d, cmap=cmap,vmin=0, vmax=vmax)
#     plt.colorbar(im, label="|∇η| (m/m)")
#     plt.title(f"SSH Gradient Magnitude\nDate: {date_str}")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.tight_layout()
#     plt.savefig(f"{figdir}/SSH_gradient_day_{date_str}.png", dpi=150)
#     plt.close()


# ======= define function for parallel plotting =======
def plot_one_frame(t):

    eta_grad_2d = eta_grad_mag.isel(time=t).compute()
    date_str = str(eta_grad_mag.time[t].values)[:10]

    fig = plt.figure(figsize=(6, 5))
    im = plt.pcolormesh(lon2d, lat2d, eta_grad_2d, cmap=cmap, vmin=0, vmax=vmax)
    plt.colorbar(im, label="|∇η| (m/m)")
    plt.title(f"SSH Gradient Magnitude\nDate: {date_str}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    outpath = f"{figdir}/SSH_gradient_day_{date_str}.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

# ======= plot all figures =======
futures = client.map(plot_one_frame, list(range(len(eta_grad_mag["time"]))))
results = client.gather(futures)



##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_gradient"
# high-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SSH_gradient-hires.mp4"
os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/SSH_gradient_day_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")
# low-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SSH_gradient-lores.mp4"
cmd = (
    f"ffmpeg -y -r 10 -pattern_type glob -i '{figdir}/SSH_gradient_day_*.png' "
    f"-vf scale=iw/2:ih/2 "
    f"-vcodec mpeg4 "
    f"-q:v 1 "
    f"-pix_fmt yuv420p "
    f"{output_movie}"
)
os.system(cmd)






# ========================================
# Compute Non-Overlapping Weekly PDFs
# ========================================
weekly_pdf_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_gradient_weekly_PDFs"
os.makedirs(weekly_pdf_dir, exist_ok=True)

pdf_data_dir = f"{output_path}/weekly_PDF_data"
os.makedirs(pdf_data_dir, exist_ok=True)

n_days = eta_grad_mag.sizes['time']
week_length = 7
n_weeks = n_days // week_length

for week_idx in range(n_weeks):
    start = week_idx * week_length
    end = start + week_length

    eta_week = eta_grad_mag.isel(time=slice(start, end))
    date_str = str(eta_week.time[0].values)[:10]  # First day of the week

    # Flatten all values from 7 days and spatial dims
    grad_flat = eta_week.values.reshape(-1)
    grad_flat = grad_flat[~np.isnan(grad_flat)]

    if grad_flat.size == 0:
        print(f"Week {date_str} has no valid data. Skipping.")
        continue

    # Define log-spaced bins
    min_val = grad_flat.min()
    max_val = grad_flat.max()
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    
    # Compute PDF
    pdf_vals, bin_edges = np.histogram(grad_flat, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Save data
    np.savez(os.path.join(pdf_data_dir, f"PDF_week_{date_str}.npz"),
             bin_centers=bin_centers,
             pdf_vals=pdf_vals)

    # Plot linear PDF
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, pdf_vals, color='black', linewidth=2)
    plt.title(f"PDF of SSH Gradient Magnitude\nWeek Starting: {date_str}")
    plt.xlabel("|∇η| (m/m)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.xscale('linear')  # Linear x-axis
    plt.yscale('log')
    plt.xlim(1e-8, 1e-4)
    plt.ylim(3, 1e6)
    plt.tight_layout()
    plt.savefig(os.path.join(weekly_pdf_dir, f"SSH_grad_PDF_week_{date_str}.png"), dpi=150)
    plt.close()

    # Plot log-log PDF
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, pdf_vals, color='black', linewidth=2)
    plt.title(f"PDF of SSH Gradient Magnitude (Log-Log)\nWeek Starting: {date_str}")
    plt.xlabel("|∇η| (m/m)")
    plt.ylabel("Probability Density")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-8, 1e-3)
    plt.ylim(3, 1e6)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(weekly_pdf_dir, f"SSH_grad_PDF_loglog_week_{date_str}.png"), dpi=150)
    plt.close()

    print(f"Saved log-binned weekly PDF and plots for week starting {date_str}")


##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_gradient_weekly_PDFs"
# high-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SSH_grad_PDF_loglog.mp4"
os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/SSH_grad_PDF_loglog_week_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")
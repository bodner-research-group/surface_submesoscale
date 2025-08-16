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
print("Loading grid...")
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

lon = ds_grid_face['XC']
lat = ds_grid_face['YC']

# ========= Load daily averaged Eta =========
print("Loading daily averaged Eta...")
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
print("Saving results...")

ds_out = xr.Dataset({
    "eta_grad_mag_daily": eta_grad_mag_daily,
    "eta_grad_mag_weekly": eta_grad_mag_weekly
})

ds_out.to_netcdf(os.path.join(output_path, "SSH_gradient_magnitude.nc"))

print("Done: daily and weekly averaged magnitude of SSH gradient saved.")








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

vmax = 2.2e-6
threshold = 1e-6

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




# Plot SSH gradient magnitude 
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_gradient"
os.makedirs(figdir, exist_ok=True)

for t in range(len(eta_grad_mag["time"])):
    eta_grad_2d = eta_grad_mag.isel(time=t)
    date_str = str(eta_grad_mag.time[t].values)[:10]

    fig = plt.figure(figsize=(6, 5))
    im = plt.pcolormesh(lon, lat, eta_grad_2d, cmap=cmap,vmin=0, vmax=vmax)
    plt.colorbar(im, label="|∇η| (m/m)")
    plt.title(f"SSH Gradient Magnitude\nDate: {date_str}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{figdir}/SSH_gradient_day_{date_str}.png", dpi=150)
    plt.close()



##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_gradient"
# high-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SSH_gradient-hires.mp4"
os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/combined_norm_map_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")
# low-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-SSH_gradient-lores.mp4"
cmd = (
    f"ffmpeg -y -r 10 -pattern_type glob -i '{figdir}/combined_norm_map_*.png' "
    f"-vf scale=iw/2:ih/2 "
    f"-vcodec mpeg4 "
    f"-q:v 1 "
    f"-pix_fmt yuv420p "
    f"{output_movie}"
)
os.system(cmd)

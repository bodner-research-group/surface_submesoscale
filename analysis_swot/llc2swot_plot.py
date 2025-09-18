import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from set_colormaps import WhiteBlueGreenYellowRed

# === Config ===
combined_nc_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/llc4320_to_swot_combined/LLC4320_on_SWOT_GRID_L3_LR_SSH_009_combined.nc"
output_images_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/figs/llc_ssh_plots_cycle009/"
output_video_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/figs/llc_ssh_cycle009.mp4"

os.makedirs(output_images_dir, exist_ok=True)

# === Load dataset ===
print("📦 Loading dataset...")
ds = xr.open_dataset(combined_nc_path)

ssh_all = np.array(ds["ssh"].values)
lat_all = np.array(ds["latitude"].values)
lon_all = np.array(ds["longitude"].values)
times = ds["swot_time"].values

num_files = ssh_all.shape[0]
cmap = WhiteBlueGreenYellowRed()

vmin = np.nanmin(ssh_all)
vmax = np.nanmax(ssh_all)
print(f"🎨 Colorbar limits: vmin = {vmin:.4f}, vmax = {vmax:.4f}")

# === Create one persistent figure ===
print("🖼️ Creating persistent figure for cumulative drawing...")
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# Keep track of plotted layers
plotted_images = []

for index in range(num_files):
    ssh = ssh_all[index, :, :]
    lat = lat_all[index, :, :]
    lon = lon_all[index, :, :]
    lon = np.where(lon > 180, lon - 360, lon)

    im = ax.pcolormesh(lon, lat, ssh,
                       transform=ccrs.PlateCarree(),
                       shading="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       alpha=0.5)

    plotted_images.append(im)

    # Add colorbar only once
    if index == 0:
        cb = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, label='SSH (m)')

    ax.set_title(f"Overlayed SSH from t=0 to t={index} hours ({np.datetime_as_string(times[index], unit='m')})")

    out_path = os.path.join(output_images_dir, f"cumulative_ssh_{index:03d}.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"✅ Saved: {out_path}")

plt.close(fig)



##### Convert images to video
import os
# high-resolution
os.system(f"ffmpeg -r 10 -pattern_type glob -i '{output_images_dir}/cumulative_ssh_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_video_path}")

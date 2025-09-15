import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client, LocalCluster
import imageio
from set_colormaps import WhiteBlueGreenYellowRed
cmap = WhiteBlueGreenYellowRed()


# === Config ===
combined_nc_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/llc4320_to_swot_combined/LLC4320_on_SWOT_GRID_L3_LR_SSH_008_combined.nc"  
output_images_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/figs/llc_ssh_plots_cycle008/"
output_video_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/figs/llc_ssh_cycle008.mp4"
n_workers = 64

os.makedirs(output_images_dir, exist_ok=True)

# === Setup Dask cluster ===
cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
client = Client(cluster)
print(f"Dask dashboard: {client.dashboard_link}")

print("Loading combined dataset...")
ds = xr.open_dataset(combined_nc_path)

# Extract ssh data (dimensions: file, num_lines, num_pixels)
ssh_all = ds["ssh"].values  # shape (num_files, num_lines, num_pixels)

# Calculate global min/max for colorbar limits (ignoring nan)
vmin = np.nanmin(ssh_all)
vmax = np.nanmax(ssh_all)
print(f"Colorbar limits: vmin={vmin:.4f}, vmax={vmax:.4f}")

lat_all = ds["latitude"].values
lon_all = ds["longitude"].values
times = ds["swot_time"].values
num_files = ssh_all.shape[0]


@dask.delayed
def plot_cumulative_ssh(index):
    """
    Plot cumulative SSH from 0 to index (inclusive) on global map.
    Save figure to disk.
    """ 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cumulative_ssh = np.nanmean(ssh_all[:index+1, :, :], axis=0)  
    cumulative_lat = np.nanmean(lat_all[:index+1, :, :], axis=0)
    cumulative_lon = np.nanmean(lon_all[:index+1, :, :], axis=0)

    # Optional: Wrap longitudes to [-180, 180] if necessary
    cumulative_lon = np.where(cumulative_lon > 180, cumulative_lon - 360, cumulative_lon)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(cumulative_lon, cumulative_lat, cumulative_ssh,
                       transform=ccrs.PlateCarree(),
                       shading="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)

    ax.coastlines()
    ax.set_global()
    ax.set_title(f"Cumulative SSH up to swot_time {np.datetime_as_string(times[index], unit='m')}")
    plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, label='SSH (m)')
    
    out_path = os.path.join(output_images_dir, f"cumulative_ssh_{index:03d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path

print("Creating delayed plot tasks...")
tasks = [plot_cumulative_ssh(i) for i in range(num_files)]
image_paths = dask.compute(*tasks)

print("All plots generated, now creating video...")

with imageio.get_writer(output_video_path, fps=5) as writer:
    for img_path in image_paths:
        image = imageio.imread(img_path)
        writer.append_data(image)

print(f"Video saved to {output_video_path}")

# Shutdown Dask client
client.shutdown()

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client, LocalCluster
import imageio

# === Config ===
combined_nc_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/llc4320_to_swot_combined/LLC4320_on_SWOT_GRID_L3_LR_SSH_008_combined.nc"  
output_images_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/ssh_plots_cycle008/"
output_video_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/ssh_evolution.mp4"
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
    Plot cumulative SSH from 0 to index (inclusive).
    Save figure to disk.
    """
    import matplotlib.pyplot as plt  # import inside function for dask workers

    cumulative_ssh = np.nanmean(ssh_all[:index+1, :, :], axis=0)  # average SSH over time steps 0 to index
    cumulative_lat = np.nanmean(lat_all[:index+1, :, :], axis=0)
    cumulative_lon = np.nanmean(lon_all[:index+1, :, :], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(cumulative_lon, cumulative_lat, cumulative_ssh, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"Cumulative SSH up to swot_time {np.datetime_as_string(times[index], unit='m')}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="SSH (m)")
    plt.tight_layout()
    out_path = os.path.join(output_images_dir, f"cumulative_ssh_{index:03d}.png")
    plt.savefig(out_path)
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

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免多线程绘图问题
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client, LocalCluster
import imageio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from set_colormaps import WhiteBlueGreenYellowRed

# === Config ===
combined_nc_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/llc4320_to_swot_combined/LLC4320_on_SWOT_GRID_L3_LR_SSH_008_combined.nc"
output_images_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/figs/llc_ssh_plots_cycle008/"
output_video_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_swot/figs/llc_ssh_cycle008.mp4"
n_workers = 16  # 可根据系统资源调整

os.makedirs(output_images_dir, exist_ok=True)

# === Setup Dask ===
cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
client = Client(cluster)
print(f"✅ Dask dashboard: {client.dashboard_link}")

# === Load dataset ===
print("📦 Loading dataset...")
ds = xr.open_dataset(combined_nc_path)

ssh_all = np.array(ds["ssh"].values)        # 强制从 Dask 转为 NumPy
lat_all = np.array(ds["latitude"].values)
lon_all = np.array(ds["longitude"].values)
times = ds["swot_time"].values

num_files = ssh_all.shape[0]
cmap = WhiteBlueGreenYellowRed()

# === Color limits ===
vmin = np.nanmin(ssh_all)
vmax = np.nanmax(ssh_all)
print(f"🎨 Colorbar limits: vmin = {vmin:.4f}, vmax = {vmax:.4f}")

# === Plotting function ===
@dask.delayed
def plot_cumulative_ssh(index):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Plot each frame up to index
        for i in range(index + 1):
            ssh = ssh_all[i, :, :]
            lat = lat_all[i, :, :]
            lon = lon_all[i, :, :]

            # Wrap longitude if needed
            lon = np.where(lon > 180, lon - 360, lon)

            im = ax.pcolormesh(lon, lat, ssh,
                               transform=ccrs.PlateCarree(),
                               shading="auto",
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               alpha=0.5)  # alpha<1 allows visual stacking

        ax.coastlines()
        ax.set_global()
        ax.set_title(f"Overlayed SSH from t=0 to t={index} ({np.datetime_as_string(times[index], unit='m')})")
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, label='SSH (m)')

        out_path = os.path.join(output_images_dir, f"cumulative_ssh_{index:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved overlay image: {out_path}")
        return out_path

    except Exception as e:
        print(f"❌ Error at index {index}: {e}")
        return None


# === Create all delayed tasks ===
print("📷 Creating plot tasks...")
tasks = [plot_cumulative_ssh(i) for i in range(num_files)]

# === Run all plotting tasks ===
print("🚀 Executing tasks...")
image_paths = dask.compute(*tasks)

# === Filter failed outputs ===
image_paths = [p for p in image_paths if p and os.path.exists(p)]
print(f"✅ {len(image_paths)} images successfully generated.")

# === Create video ===
print("🎞️ Creating video...")
with imageio.get_writer(output_video_path, fps=5) as writer:
    for img_path in sorted(image_paths):
        image = imageio.imread(img_path)
        writer.append_data(image)

print(f"🎉 Video saved to: {output_video_path}")

# === Shutdown ===
client.shutdown()

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
from my_colormaps import WhiteBlueGreenYellowRed

from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

cmap = WhiteBlueGreenYellowRed()

# ==== Constants ====
omega = 7.2921e-5  # [rad/s]

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# ==== Paths ====
data_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/strain_vorticity/strain_vorticity_daily.nc"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/strain_vorticity"
os.makedirs(figdir, exist_ok=True)

# ==== Load data ====
ds = xr.open_dataset(data_path)

# Load the model
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
lat_g = ds1.YG.isel(face=face,i_g=1,j_g=j)
lon_g = ds1.XG.isel(face=face,i_g=i,j_g=1)

# Convert lat/lon from xarray to NumPy arrays
lat_vals = lat.values  # shape (j,)
lon_vals = lon.values  # shape (i,)
lat_g_vals = lat_g.values  # shape (j,)
lon_g_vals = lon_g.values  # shape (i,)

# Create 2D lat/lon meshgrid
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')  # shape (j, i)
lon_g_2d, lat_g_2d = np.meshgrid(lon_g_vals, lat_g_vals, indexing='xy')  # shape (j, i)

# ==== Compute Coriolis parameter (mean) ====
lat_rad = np.deg2rad(lat)
f0 = 2 * omega * np.sin(lat_rad)
f0_mean = f0.mean().compute()
print(f"Mean f0 over domain: {f0_mean:.2e} s^-1")

# ==== Normalize variables ====
sigma_norm = ds["strain_mag"] / abs(f0_mean)
zeta_norm = ds["vorticity"] / f0_mean
delta_norm = ds["divergence"] / f0_mean

# ==== Daily Maps ====
print("\nGenerating daily maps...")
for t in range(len(ds.time)):
    date_str = str(ds.time[t].values)[:10]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    # Data for current time step
    v0 = zeta_norm.isel(time=t)
    v1 = sigma_norm.isel(time=t)
    v2 = delta_norm.isel(time=t)

    # Add date as figure-level title
    fig.suptitle(f"Date: {date_str}", fontsize=17)

    # subplot 1
    im0 = axes[0].pcolormesh(lon_g_2d, lat_g_2d, v0, cmap='RdBu_r', shading="auto", vmin=-1, vmax=1)
    axes[0].set_title(r'$\zeta/f_0$')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    cbar0 = plt.colorbar(im0, ax=axes[0], orientation="vertical", pad=0.02)
    cbar0.set_label(r'$\zeta/f_0$')

    # subplot 2
    im1 = axes[1].pcolormesh(lon2d, lat2d, v1, cmap='viridis', shading="auto", vmin=0, vmax=1)
    axes[1].set_title(r'$\sigma/|f_0|$')
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    cbar1 = plt.colorbar(im1, ax=axes[1], orientation="vertical", pad=0.02)
    cbar1.set_label(r'$\sigma/|f_0|$')

    # subplot 3
    im2 = axes[2].pcolormesh(lon2d, lat2d, v2, cmap='BrBG', shading="auto", vmin=-1, vmax=1)
    axes[2].set_title(r'$\Delta/f_0$')
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    cbar2 = plt.colorbar(im2, ax=axes[2], orientation="vertical", pad=0.02)
    cbar2.set_label(r'$\Delta/f_0$')

    # Save figure
    outpath = os.path.join(figdir, f"combined_norm_map_{date_str}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()

    if t % 10 == 0:
        print(f"  Processed {t+1}/{len(ds.time)} days")


# ==== Weekly Joint PDFs ====
print("\nGenerating weekly joint PDFs...")
n_days = len(ds.time)
days_per_week = 7
n_weeks = n_days // days_per_week

bins = 300
limit_value = 1.2
sigma_range = (0, limit_value)
zeta_range = (-limit_value, limit_value)


for w in range(n_weeks):
    t0 = w * days_per_week
    t1 = t0 + days_per_week
    week_time = ds.time[t0].values

    # Select data
    sigma_w = sigma_norm.isel(time=slice(t0, t1)).values.ravel()
    zeta_w = zeta_norm.isel(time=slice(t0, t1)).values.ravel()

    # Clean
    valid = np.isfinite(sigma_w) & np.isfinite(zeta_w)
    sigma_flat = sigma_w[valid]
    zeta_flat = zeta_w[valid]

    # Histogram
    H, xedges, yedges = np.histogram2d(zeta_flat, sigma_flat,  
                                   bins=bins,
                                   range=[zeta_range, sigma_range],
                                   density=True)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    norm = colors.LogNorm(vmin=5e-2, vmax=20)

    plt.pcolormesh(X, Y, H, shading='auto', cmap=cmap, norm=norm)

    # Add grid lines and reference lines
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.3,linewidth=0.5)
    xy_min = max(min(zeta_range[0], sigma_range[0]), -1e3)
    xy_max = min(max(zeta_range[1], sigma_range[1]), 1e3)
    x_line = np.linspace(xy_min, xy_max, 100)

    plt.plot(x_line, x_line, 'k--', linewidth=1, alpha=0.7, label='$x=y$')     # x=y
    plt.plot(x_line, -x_line, 'k--', linewidth=1, alpha=0.7, label='$x=-y$')   # x=-y
    plt.ylim(bottom=0, top=limit_value)
    plt.xlim(-limit_value, limit_value)

    plt.ylabel(r"$\sigma/|f_0|$")
    plt.xlabel(r"$\zeta/f_0$")
    plt.title(f"Surface strain-vorticity JPDF ({str(week_time)[:10]})")
    plt.colorbar(label="Joint PDF")
    plt.tight_layout()

    outname = os.path.join(figdir, f"joint_pdf_sigma_zeta_week{w+1:02d}.png")
    plt.savefig(outname, dpi=200)
    plt.close()

    if w % 5 == 0:
        print(f"  Processed week {w+1}/{n_weeks}")

print("\nAll plots saved.")



##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/strain_vorticity"
# high-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-strain_vorticity-hires.mp4"
os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/combined_norm_map_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")
# low-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-strain_vorticity-lores.mp4"
cmd = (
    f"ffmpeg -y -r 10 -pattern_type glob -i '{figdir}/combined_norm_map_*.png' "
    f"-vf scale=iw/2:ih/2 "
    f"-vcodec mpeg4 "
    f"-q:v 1 "
    f"-pix_fmt yuv420p "
    f"{output_movie}"
)
os.system(cmd)


##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/strain_vorticity"
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-jointPDF.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/joint_pdf_sigma_zeta_week*.png' -vf scale=iw/2:ih/2  -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

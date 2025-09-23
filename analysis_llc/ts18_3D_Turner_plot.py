import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from set_constant import domain_name, face, i, j  
from set_colormaps import WhiteBlueGreenYellowRed
cmap = WhiteBlueGreenYellowRed()

# Path to saved output file from previous script
nc_file = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_3D/Turner_3D_7d_2011-11-01.nc"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
os.makedirs(figdir, exist_ok=True)

# Global font size setting for figures
plt.rcParams.update({'font.size': 14})

# Open dataset
ds = xr.open_dataset(nc_file)

# Extract variables
deta_cross = ds['deta_cross']
dt_cross = ds['dt_cross']
ds_cross = ds['ds_cross']
TuV_deg = ds['TuV_deg']
TuH_deg = ds['TuH_deg']
alpha = ds['alpha_surf']
beta = ds['beta_surf']
lon = ds['lon2d']
lat = ds['lat2d']

# Make masks to exclude NaNs
mask = np.isfinite(deta_cross) & np.isfinite(dt_cross) & np.isfinite(ds_cross)

# ============
# (1) Map: deta_cross, dt_cross, ds_cross
# ============
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

# Prepare lon/lat for plotting
lon_plot = lon.transpose("j", "i").values
lat_plot = lat.transpose("j", "i").values

# Crop lon/lat for shading='auto' in pcolormesh
lon_plot = lon_plot[:-1, :-1]
lat_plot = lat_plot[:-1, :-1]

# Crop data variables
deta_plot = deta_cross[:-1, :-1]
dt_plot = dt_cross[:-1, :-1]
ds_plot = ds_cross[:-1, :-1]

# Compute symmetric color limits
vmax_deta = np.nanmax(np.abs(deta_plot))
vmax_dt = np.nanmax(np.abs(dt_plot))
vmax_ds = np.nanmax(np.abs(ds_plot))

# Create 3-panel figure
fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)

# Panel 1: deta_cross
pcm0 = axs[0].pcolormesh(
    lon_plot, lat_plot, deta_plot, cmap='RdBu_r', shading='auto',
    vmin=-vmax_deta, vmax=vmax_deta
)
axs[0].set_title('Cross-Isopycnal SSH Gradient')
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")
fig.colorbar(pcm0, ax=axs[0], orientation='vertical')

# Panel 2: dt_cross
pcm1 = axs[1].pcolormesh(
    lon_plot, lat_plot, dt_plot, cmap='RdBu_r', shading='auto',
    vmin=-vmax_dt, vmax=vmax_dt
)
axs[1].set_title('Cross-Isopycnal Temperature Gradient')
axs[1].set_xlabel("Longitude")
axs[1].set_ylabel("Latitude")
fig.colorbar(pcm1, ax=axs[1], orientation='vertical')

# Panel 3: ds_cross
pcm2 = axs[2].pcolormesh(
    lon_plot, lat_plot, ds_plot, cmap='RdBu_r', shading='auto',
    vmin=-vmax_ds, vmax=vmax_ds
)
axs[2].set_title('Cross-Isopycnal Salinity Gradient')
axs[2].set_xlabel("Longitude")
axs[2].set_ylabel("Latitude")
fig.colorbar(pcm2, ax=axs[2], orientation='vertical')

# Save the combined figure
fig_path = os.path.join(figdir, "cross_gradients_combined_map.png")
plt.savefig(fig_path, dpi=300)
plt.close()



# ============
# (2) Map: TuV_deg and TuH_deg
# ============
# Prepare lon/lat for plotting
lon_plot = lon.transpose("j", "i").values
lat_plot = lat.transpose("j", "i").values
lon_plot = lon_plot[:-1, :-1]
lat_plot = lat_plot[:-1, :-1]

# Prepare data
TuV_plot = TuV_deg[:-1, :-1]
TuH_plot = TuH_deg[:-1, :-1]
Tu_diff_plot = TuV_plot - TuH_plot

# Set consistent color limits
angle_vmin, angle_vmax = -180, 180
diff_vmax = np.nanmax(np.abs(Tu_diff_plot))  # For centered color scale
diff_vmax = min(diff_vmax, 360)  # Limit to a reasonable range if needed

# Create 3-panel figure
fig, axs = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

# Panel 1: TuV_deg
pcm0 = axs[0].pcolormesh(
    lon_plot, lat_plot, TuV_plot,
    cmap='twilight', shading='auto',
    vmin=angle_vmin, vmax=angle_vmax
)
axs[0].set_title('Vertical Turner Angle (TuV)')
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")
fig.colorbar(pcm0, ax=axs[0], orientation='vertical')

# Panel 2: TuH_deg
pcm1 = axs[1].pcolormesh(
    lon_plot, lat_plot, TuH_plot,
    cmap='twilight', shading='auto',
    vmin=angle_vmin, vmax=angle_vmax
)
axs[1].set_title('Horizontal Turner Angle (TuH)')
axs[1].set_xlabel("Longitude")
axs[1].set_ylabel("Latitude")
fig.colorbar(pcm1, ax=axs[1], orientation='vertical')

# Panel 3: TuV_deg - TuH_deg
pcm2 = axs[2].pcolormesh(
    lon_plot, lat_plot, Tu_diff_plot,
    cmap='coolwarm', shading='auto',
    vmin=-diff_vmax, vmax=diff_vmax
)
axs[2].set_title('Turner Angle Difference (TuV - TuH)')
axs[2].set_xlabel("Longitude")
axs[2].set_ylabel("Latitude")
fig.colorbar(pcm2, ax=axs[2], orientation='vertical')

# Save combined figure
fig_path = os.path.join(figdir, "turner_angles_combined_map.png")
plt.savefig(fig_path, dpi=300)
plt.close()




# ============
# (3) 3D Scatter: deta_cross (z), beta*ds_cross (x), alpha*dt_cross (y)
# ============
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = (beta * ds_cross).values[mask]
y = (alpha * dt_cross).values[mask]
z = deta_cross.values[mask]

sc = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, s=1)
ax.set_xlabel('β·dS_cross')
ax.set_ylabel('α·dT_cross')
ax.set_zlabel('deta_cross')
fig.colorbar(sc, label='deta_cross')
ax.set_title("3D Scatter: Cross-Isopycnal SSH Gradient")
fig_path = os.path.join(figdir, "3d_scatter_deta_cross.png")
plt.savefig(fig_path, dpi=300)
plt.close()


from scipy.io import savemat

# Prepare data for saving
scatter_data = {
    'x': (beta * ds_cross).values[mask],
    'y': (alpha * dt_cross).values[mask],
    'z': deta_cross.values[mask]
}

# Save as MATLAB .mat file
mat_path = os.path.join(figdir, "3d_scatter_deta_cross.mat")
savemat(mat_path, scatter_data)



# ============
# (4) Histogram and Kernel PDF of deta_cross
# ============
deta_clean = deta_cross.values[mask]
fig, ax = plt.subplots(figsize=(8.5, 5))

sns.histplot(deta_clean, kde=True, bins=73, color='skyblue', stat='density', edgecolor='none')

ax.set_title("Histogram and KDE of Cross-Isopycnal SSH Gradient")
ax.set_xlabel(r'$|\partial \eta|$')
ax.set_ylabel("Density")

# Enable minor ticks and grid
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.4)

fig_path = os.path.join(figdir, "hist_kde_deta_cross.png")
plt.savefig(fig_path, dpi=300)
plt.close()


# ============
# (5) Bin TuH_deg, compute mean |deta_cross| per bin
# ============
TuH_vals = TuH_deg.values[mask]
deta_vals = np.abs(deta_cross.values[mask])

# Define bins and compute digitized bin indices
bins = np.linspace(-180, 180, 73)  # 5-degree bins
bin_centers = 0.5 * (bins[:-1] + bins[1:])
digitized = np.digitize(TuH_vals, bins) - 1

# Compute mean |deta_cross| in each bin
mean_deta_per_bin = np.array([
    deta_vals[digitized == i].mean() if np.any(digitized == i) else np.nan
    for i in range(len(bin_centers))
])

# Plot result
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bin_centers, mean_deta_per_bin, marker='o', linestyle='-')
ax.set_xlabel("Horizontal Turner Angle (deg)")
ax.set_ylabel(r"Mean $|\partial \eta|$")
ax.set_title("Mean Cross-Isopycnal SSH Gradient vs. Turner Angle")
ax.grid(True)
fig_path = os.path.join(figdir, "turner_angle_vs_deta_cross.png")
plt.savefig(fig_path, dpi=300)
plt.close()




# ============
# (6) 3D Plot: PDF Vector (β·∂S, α·∂θ) vs. Mean |∂η| per TuH bin with Turner PDF vectors on z=0 plane
# ============

# Required variables:
# - bin_centers (already defined from np.linspace(-180, 180, 37))
# - mean_deta_per_bin (computed from TuH_vals vs. |deta_cross|)
# - x_grid, pdf_values_h, pdf_values_v, beta, alpha (from original dataset)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# Required inputs:
# - bin_centers: angle bin centers (e.g., np.linspace(-180, 180, 37))
# - mean_deta_per_bin: mean |∂η| per bin (same length as bin_centers)
# - nc_file: path to NetCDF dataset
# - figdir: directory to save figures

scale = 10000  # scaling factor for vector lengths

# Open dataset
ds_pdf = xr.open_dataset(nc_file)

x_grid = ds_pdf["x_grid"].data  # angle bins in degrees
pdf_values_h = ds_pdf["pdf_values_h"].data
pdf_values_v = ds_pdf["pdf_values_v"].data
alpha = np.nanmean(ds_pdf["alpha_surf"].data)
beta = np.nanmean(ds_pdf["beta_surf"].data)

# Define unit vectors for projection space
slope_rho = 1
v_cross = np.array([-slope_rho, 1.0])
v_iso = np.array([1.0, slope_rho])
v_cross /= np.linalg.norm(v_cross)
v_iso /= np.linalg.norm(v_iso)

# Interpolate mean |∂η| onto x_grid (PDF bins)
valid = ~np.isnan(mean_deta_per_bin)
interp_func = interp1d(bin_centers[valid], mean_deta_per_bin[valid],
                       kind='linear', bounds_error=False, fill_value=np.nan)
z_vals = interp_func(x_grid)

# Compute x_proj, y_proj using the same formula/scaling as horizontal PDF vectors plot
x_proj = []
y_proj = []

for angle_deg, mag_h in zip(x_grid, pdf_values_h):
    dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
    dx = mag_h * dir_vec[0] * beta * scale * 20  # same scaling as horizontal vectors
    dy = mag_h * dir_vec[1] * alpha * scale * 20
    x_proj.append(dx)
    y_proj.append(dy)

# Prepare plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# # Normalize z_vals for colormap (handle NaNs carefully)
# z_min = np.nanmin(z_vals)
# z_max = np.nanmax(z_vals)
# normed_vals = (z_vals - z_min) / (z_max - z_min)

# # Plot 3D lines from origin to (x_proj, y_proj, z_vals)
# for x, y, z, cval in zip(x_proj, y_proj, z_vals, normed_vals):
#     if not np.isnan(z):
#         ax.plot([0, x], [0, y], [0, z], color=plt.cm.plasma(cval), alpha=0.8, linewidth=1.0)

# # Dummy scatter to enable colorbar
# sc = ax.scatter([0], [0], [0], c=[z_min], cmap=cmap, vmin=0, vmax=6e-6)

# Normalization limits
z_min_plot = 0
z_max_plot = 6e-6

# Clip and normalize z_vals to [0,6e-6] for coloring
normed_vals = np.clip(z_vals, z_min_plot, z_max_plot)
normed_vals = (normed_vals - z_min_plot) / (z_max_plot - z_min_plot)

# Plot 3D lines from origin to (x_proj, y_proj, z_vals) with custom cmap
for x, y, z, cval in zip(x_proj, y_proj, z_vals, normed_vals):
    if not np.isnan(z):
        ax.plot([0, x], [0, y], [0, z], color=cmap(cval), alpha=0.8, linewidth=1.0)

# Dummy scatter to enable colorbar using same cmap
sc = ax.scatter([0], [0], [0], c=[0], cmap=cmap, vmin=z_min_plot, vmax=z_max_plot)

# Plot TuH (horizontal PDF vectors) at z=0 plane
for angle_deg, mag_h in zip(x_grid, pdf_values_h):
    dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
    x0, y0, z0 = 0, 0, 0
    dx = mag_h * dir_vec[0] * beta * scale * 20
    dy = mag_h * dir_vec[1] * alpha * scale * 20
    ax.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0], color='darkgray', linestyle='--', alpha=0.7, linewidth=0.7)

# Plot TuV (vertical PDF vectors) at z=0 plane
for angle_deg, mag_v in zip(x_grid, pdf_values_v):
    dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
    x0, y0, z0 = 0, 0, 0
    dx = mag_v * dir_vec[0] * beta * scale * 10
    dy = mag_v * dir_vec[1] * alpha * scale * 10
    ax.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0], color='green', alpha=0.7, linewidth=0.7)

# Set axis labels and title
ax.set_xlabel(r"$\beta \, \partial S$ (PDF projection)")
ax.set_ylabel(r"$\alpha \, \partial \theta$ (PDF projection)")
ax.set_zlabel(r"Mean $|\partial \eta|$")
ax.set_title("3D Turner PDF Vectors vs. SSH Gradient")

# Apply axis limits
ax.set_xlim([-1, 4])
ax.set_ylim([-1, 6])
ax.set_zlim([0, 6e-6])

# Add colorbar
cbar = fig.colorbar(sc, ax=ax, label=r"Mean $|\partial \eta|$", shrink=0.5, fraction=0.02, pad=0.05)

# Save figure
fig_path = os.path.join(figdir, "3d_turner_pdf_with_lines.png")
plt.savefig(fig_path, dpi=300)
plt.close()
ds_pdf.close()

print("✅ 3D Turner PDF with lines from origin to horizontal PDF vectors added.")

# Calculate the vertical Turner Angle following Johnson et al. (2016) JPO
# Using instantaneous output or time-averages

# 1. Plot surface T and S
# 2. Compute potential density, alpha, and beta
# 3. Compute and plot the mixed layer depth
# 4. Find 50%-90% of the mixed layer
# 5. Compute vertical gradients of temperature and salinity
# 6. Compute the Turner angle for vertical gradients
# 7. Plot surface U, V, and W
# 8. Calculate the Kernel PDF distribution


# Load packages
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import zarr 
import dask 
import gsw  # (Gibbs Seawater Oceanography Toolkit) https://teos-10.github.io/GSW-Python/gsw.html

from numpy.linalg import lstsq
import seaborn as sns
from scipy.stats import gaussian_kde

from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import pandas as pd


# Load the model
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Folder to store the figures
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_test2_day1_3pm"

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# Set spatial indices
face = 1
k_surf = 0
# i = slice(0,100,1) # Southern Ocean
# j = slice(0,101,1) # Southern Ocean
i = slice(1000,1200,1) # Tropics
j = slice(2800,3001,1) # Tropics

# Grid spacings in m
dxF = ds1.dxF.isel(face=face,i=i,j=j)
dyF = ds1.dyF.isel(face=face,i=i,j=j)

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
depth = ds1.Z

# Convert lat/lon from xarray to NumPy arrays
lat_vals = lat.values  # shape (j,)
lon_vals = lon.values  # shape (i,)

# Create 2D lat/lon meshgrid
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')  # shape (j, i)

# Find the center location of selected region
lat_c = float(lat.mean().values)
lon_c = float(lon.mean().values)
# print(f"Center location: lat={lat_c}, lon={lon_c}")

# Find the time zone
tf = TimezoneFinder()
timezone_str = tf.timezone_at(lng=lon_c, lat=lat_c)
# print(f"Detected timezone: {timezone_str}")

# Convert UTC to Local time
time_utc = pd.to_datetime(ds1.time.values)
time_local = time_utc.tz_localize('UTC').tz_convert(timezone_str)
# print(time_local[:5])

# Set temporal indices:
indices_14 = [time_idx for time_idx, t in enumerate(time_local) if t.hour == 14]  # 2pm local time
indices_15 = [time_idx for time_idx, t in enumerate(time_local) if t.hour == 15]  # 3pm local time
indices_16 = [time_idx for time_idx, t in enumerate(time_local) if t.hour == 16]  # 4pm local time

# print(indices_15)
# print(len(indices_15))

time_inst = indices_15[0]   

nday_avg = 7                 # 7-day average
# time_avg = slice(0,24*nday_avg,1)  
time_avg = []
for time_idx in range(nday_avg):
    time_avg.extend([indices_14[time_idx], indices_15[time_idx], indices_16[time_idx]])


##############################################################
### 0. Load instantaneous output or compute time averages  ###
##############################################################

################ If use instantaneous output
tt = ds1.Theta.isel(time=time_inst,face=face,i=i,j=j) # Potential temperature, shape (k, j, i)
ss = ds1.Salt.isel(time=time_inst,face=face,i=i,j=j)  # Practical salinity, shape (k, j, i)
################ End if use instantaneous output


# ################ If use time averages
# # Read temperature and salinity data of the top 1000 m 
# tt = ds1.Theta.isel(time=time_avg,face=face,i=i,j=j) # Potential temperature
# ss = ds1.Salt.isel(time=time_avg,face=face,i=i,j=j)  # Practical salinity
# # eta = ds1.Eta.isel(time=time,face=face,i=i,j=j)  # Surface Height Anomaly
# print(tt.chunks) 

# # # Re-chunk time dimension for efficient averaging
# # tt = tt.chunk({'time': 24*nday_avg})   # Re-chunk to 7-day blocks
# # ss = ss.chunk({'time': 24*nday_avg})   # Re-chunk to 7-day blocks
# tt = tt.chunk({'time': -1})  # Re-chunk to include all data points
# ss = ss.chunk({'time': -1})  # Re-chunk to include all data points
# print(tt.chunks) 

# # Compute time averages
# # Build a lazy Dask graph — nothing is computed yet
# tt_mean = tt.mean(dim='time')
# ss_mean = ss.mean(dim='time')

# # Trigger computation
# tt = tt_mean.compute()
# ss = ss_mean.compute()
# ################ End if use time averages


###############################
### 1. Plot surface T and S ###
###############################

tt_surf = tt.isel(k=k_surf)
ss_surf = ss.isel(k=k_surf)

fig, axs = plt.subplots(1,2,figsize=(15,5))

pcm_t = axs[0].pcolormesh(lon2d, lat2d, tt_surf, shading='auto', cmap='gist_ncar')
axs[0].set_title('Surface Temperature')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
fig.colorbar(pcm_t, ax=axs[0], label='(\u00B0C)')
# cbar_t = fig.colorbar(pcm_t, ax=axs[0])
# cbar_t.ax.set_title('(\u00B0C)')  # Label on top

pcm_s = axs[1].pcolormesh(lon2d, lat2d, ss_surf, shading='auto', cmap='terrain')
axs[1].set_title('Surface Salinity')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
fig.colorbar(pcm_s, ax=axs[1], label='(psu)')
# cbar_s = fig.colorbar(pcm_s, ax=axs[1])
# cbar_s.ax.set_title('(psu)')      # Label on top

plt.tight_layout()
plt.savefig(f"{figdir}/surface_t_s_200.png", dpi=150)
plt.close()


#####################################################
### 2. Compute potential density, alpha, and beta ###
#####################################################
### Note: Ideally we should compute potential density, alpha, and beta every time step, and then take the time average
### However, for an estimation, here we use the time-averaged T and S to compute density (if time-averaged tt and ss were used)

# Compute the potential density using GSW (Gibbs Seawater Oceanography Toolkit), with surface reference pressure 
SA = gsw.conversions.SA_from_SP(ss, depth, lon, lat) # Absolute salinity, shape (k, j, i)
CT = gsw.conversions.CT_from_pt(SA, tt)              # Conservative temperature, shape (k, j, i)
p_ref = 0                                            # Reference pressure 
rho = gsw.density.rho(SA, CT, p_ref)                 # Potential density, shape (k, j, i)
alpha = gsw.density.alpha(SA, CT, depth)             # Thermal expansion coefficient with respect to Conservative Temperature, shape (k, j, i)
beta = gsw.density.beta(SA, CT, depth)               # Saline (i.e. haline) contraction coefficient of seawater at constant Conservative Temperature, shape (k, j, i)

# # Plot to check potential density
# fig, axs = plt.subplots(1,3,figsize=(22,5))
# rho.isel(k=0).plot(ax=axs[0], vmin=1027.25,vmax=1027.6,cmap="terrain",add_labels=False)
# rho.isel(k=25).plot(ax=axs[1],vmin=1027.25,vmax=1027.6,cmap="terrain",add_labels=False)
# rho.isel(k=35).plot(ax=axs[2],vmin=1027.25,vmax=1027.6,cmap="terrain",add_labels=False)
# plt.savefig(f"{figdir}/test_density.png", dpi=150)
# plt.close()

# print(depth.values[35])
# print(depth.values[25])


#################################################
### 3. Compute and plot the mixed layer depth ###
#################################################

# Compute the mixed layer depth, using the delta_rho = 0.03 kg/m^2 criterion
rho_surface = rho.isel(k=0)
drho = rho - rho_surface.expand_dims({'k': rho.k})

# thresh = xr.where(np.abs(drho) > 0.03, drho.k, np.nan) # thresh = xr.where((np.abs(drho) > 0.03) & (N2 > 1e-5), drho.depth, np.nan)
# mld_idx = thresh.min("k")  # vertical indices of mixed layer depth 
# mld_idx.name = "Vertical indices of mixed layer base"

drho = drho.assign_coords(k=depth) # replace coordinate k with depth
thresh = xr.where(np.abs(drho) > 0.03, drho.k, np.nan)
mld = thresh.max("k")      # mixed layer depth, shape (j, i)
mld.name = "MLD"

# Plot the mixed layer depth indices
plt.figure(figsize=(10, 6))
pcm = plt.pcolormesh(lon2d, lat2d, mld, cmap='terrain', shading='auto')
plt.colorbar(pcm, label="(m)")
plt.title("Mixed Layed Depth")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(f"{figdir}/test_mld.png", dpi=150)
plt.close()

##########################################
### 4. Find 50%-90% of the mixed layer ###
##########################################

# Find 50%-90% of the mixed layer (Johnson et al. 2016)
mld_50 = mld * 0.5
mld_90 = mld * 0.9
dz_mld = mld_50 - mld_90     # shape (j, i)

# Expand dimensions of depth to match data shape (k, i, j)
depth_broadcasted = depth.values[:, None, None]  # (k, 1, 1)
mld_50_broadcasted = mld_50.values[None, :, :]   # (1, j, i)
mld_90_broadcasted = mld_90.values[None, :, :]   # (1, j, i)

# Find index of depth level closest to mld_50 and mld_90
abs_diff_50 = np.abs(depth_broadcasted - mld_50_broadcasted)  # (k, j, i)
abs_diff_90 = np.abs(depth_broadcasted - mld_90_broadcasted)  # (k, j, i)
k_50 = abs_diff_50.argmin(axis=0)  # shape (j, i)
k_90 = abs_diff_90.argmin(axis=0)  # shape (j, i)

# Use these indices to get the closest depth levels
depth_50 = depth.values[k_50]  # shape (j, i)
depth_90 = depth.values[k_90]  # shape (j, i)
dz_5090 = depth_50 - depth_90

# Extract temperature and salinity data at 50% and 90% of the mixed layer depth
# Convert temperature and salinity to DataArray with NumPy backing
tt_np = tt.values       # shape (k, j, i)
ss_np = ss.values       # shape (k, j, i)
alpha_np = alpha.values # shape (k, j, i) 
beta_np = beta.values   # shape (k, j, i)

# Get dimensions
j_dim, i_dim = k_50.shape

# Prepare index arrays for advanced indexing
j_idx, i_idx = np.meshgrid(np.arange(j_dim), np.arange(i_dim), indexing='ij')

# Extract temperature and salinity at k_50 and k_90
tt_k50 = tt_np[k_50, j_idx, i_idx]
tt_k90 = tt_np[k_90, j_idx, i_idx]

ss_k50 = ss_np[k_50, j_idx, i_idx]
ss_k90 = ss_np[k_90, j_idx, i_idx]

alpha_k50 = alpha_np[k_50, j_idx, i_idx]
alpha_k90 = alpha_np[k_90, j_idx, i_idx]

beta_k50 = beta_np[k_50, j_idx, i_idx]
beta_k90 = beta_np[k_90, j_idx, i_idx]


#################################################################
### 5. Compute vertical gradients of temperature and salinity ###
#################################################################


# Gradients
dT_dz = (tt_k50 - tt_k90) / dz_5090
dS_dz = (ss_k50 - ss_k90) / dz_5090
x = beta_k50 * dS_dz
y = alpha_k50 * dT_dz

# Flatten and mask NaNs
x_flat = x.ravel()
y_flat = y.ravel()
mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)

x_valid = x_flat[mask]
y_valid = y_flat[mask]

# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(x_valid, y_valid, s=10, alpha=0.4, c='tab:green')
# Compute line range
min_val = min(x_valid.min(), y_valid.min())
max_val = max(x_valid.max(), y_valid.max())

# Plot x = y line
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(r"$\beta \cdot \frac{\partial S}{\partial z}$")
plt.ylabel(r"$\alpha \cdot \frac{\partial T}{\partial z}$")
plt.title(r"Scaled Vertical Gradients: $\alpha \frac{dT}{dz}$ vs $\beta \frac{dS}{dz}$")

# Turn on minor ticks and grid
plt.minorticks_on()
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

# Legend and layout
plt.tight_layout()
plt.savefig(f"{figdir}/test_ts_gradients.png", dpi=150)
plt.close()


##########################################################
### 6. Compute the Turner angle for vertical gradients ###
##########################################################

Tu_rad = np.arctan2(y+x, y-x)  # arctan2 handles 4-quadrant correctly
Tu_deg = np.degrees(Tu_rad)

### 6.1 Plot Histogram of Turner Angle (Tu)

# Now you can scatter plot Tu, or categorize regimes:
Tu_valid = Tu_deg.ravel()[~np.isnan(Tu_deg.ravel())]
plt.figure(figsize=(7, 5))
plt.hist(Tu_valid, bins=50, color='steelblue', edgecolor='k', alpha=0.7)

# Add instability boundaries
plt.axvline(-90, color='darkred', linestyle='--', linewidth=2, label='Unstable boundary (-90°)')
plt.axvline(90, color='darkred', linestyle='--', linewidth=2, label='Unstable boundary (+90°)')

# Mark salt fingering / diffusive convection regions
plt.axvline(45, color='red', linestyle='--', label='Salt fingering limit')
plt.axvline(-45, color='green', linestyle='--', label='Diffusive convection limit')

# Labels
plt.xlabel("Turner angle (degrees)")
plt.ylabel("Count")
plt.title("Histogram of Turner Angle (Tu)")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(f"{figdir}/turner_angle_histogram.png", dpi=150)
plt.close()


### 6.2 Plot a map of the Turner Angle (Tu)

# Plot
plt.figure(figsize=(10, 6))
pcm = plt.pcolormesh(lon2d, lat2d, Tu_deg, cmap='twilight', shading='auto', vmin=-90, vmax=90)
plt.colorbar(pcm, label="Turner angle (°)")
plt.title("Turner Angle (Tu) at 50–90% MLD")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(f"{figdir}/turner_angle_map.png", dpi=150)
plt.close()



### 6.3 Plot a map of the unstable regimes based on the Turner Angle (Tu)

# Initialize regime map with NaN
regime = np.full_like(Tu_deg, np.nan)

# Fill with categories
regime[Tu_deg < -90] = 0   # Overturning unstable (cold fresh below)
regime[(Tu_deg >= -90) & (Tu_deg < -45)] = 1  # Diffusive convection
regime[(Tu_deg >= -45) & (Tu_deg <= 45)] = 2  # Doubly stable
regime[(Tu_deg > 45) & (Tu_deg <= 90)] = 3    # Salt fingering
regime[Tu_deg > 90] = 4   # Overturning unstable (warm salty above)

from matplotlib.colors import ListedColormap, BoundaryNorm

# Define labels and colors
labels = [
    'Overturning (Tu < -90)',
    'Diffusive convection',
    'Doubly stable',
    'Salt fingering',
    'Overturning (Tu > 90)'
]

colors = [
    '#8b0000',  # deep red
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8b0000'   # deep red again
]

cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)


plt.figure(figsize=(12, 6))
pcm = plt.pcolormesh(lon2d, lat2d, regime, cmap=cmap, norm=norm, shading='auto')
cbar = plt.colorbar(pcm, ticks=range(5))
cbar.ax.set_yticklabels(labels)
plt.title("Vertical Stability Regimes by Turner Angle")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(f"{figdir}/turner_angle_regimes.png", dpi=150)
plt.close()




###############################
### 7. Plot surface U, V, W ###
###############################

# Read surface velocity data
uu_surf = ds1.U.isel(time=time_inst,k=k_surf,face=face,i_g=i,j=j)
vv_surf = ds1.V.isel(time=time_inst,k=k_surf,face=face,i=i,j_g=j)
ww_surf = ds1.W.isel(time=time_inst,k_p1=k_surf,face=face,i=i,j=j)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# u component
pcm_u = axs[0].pcolormesh(lon2d, lat2d, uu_surf, shading='auto', cmap='RdBu_r')
axs[0].set_title('Surface Zonal Velocity (u)')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
fig.colorbar(pcm_u, ax=axs[0], label='m/s')

# v component
pcm_v = axs[1].pcolormesh(lon2d, lat2d, vv_surf, shading='auto', cmap='RdBu_r')
axs[1].set_title('Surface Meridional Velocity (v)')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
fig.colorbar(pcm_v, ax=axs[1], label='m/s')

# w component
pcm_w = axs[2].pcolormesh(lon2d, lat2d, ww_surf, shading='auto', cmap='RdBu_r')
axs[2].set_title('Surface Vertical Velocity (w)')
axs[2].set_xlabel('Longitude')
axs[2].set_ylabel('Latitude')
fig.colorbar(pcm_w, ax=axs[2], label='m/s')

plt.tight_layout()
plt.savefig(f"{figdir}/surface_uvw.png", dpi=150)
plt.close()






#################################################
### 8. Calculate the Kernel PDF distribution  ###
#################################################

# 1). Flatten the Tu_H_deg array and remove NaN values
Tu_V_flat = Tu_deg.flatten()
Tu_V_clean = Tu_V_flat[~np.isnan(Tu_V_flat)]  # remove NaNs

# 2). Plot Kernel PDF
plt.figure(figsize=(8, 5))
sns.kdeplot(Tu_V_clean, bw_adjust=0.5, fill=True, color="darkblue", label=r'$Tu_H$ Kernel density estimation')
plt.xlabel(r'$Tu_V$ (°)')
plt.ylabel("Density")
plt.title("Kernel PDF of Vertical Turner Angle")

plt.grid(True, which='major', linestyle=':')  # major grid lines
plt.minorticks_on()                           # enable minor ticks
plt.grid(True, which='minor', linestyle='--', alpha=0.15)  # minor grid lines (lighter and dashed)

# Highlight vertical lines at -90 and 90 degrees
plt.axvline(x=-90, color='red', linestyle='--', linewidth=2, label='-90°')
plt.axvline(x=90, color='red', linestyle='--', linewidth=2, label='90°')

plt.legend()
plt.tight_layout()
plt.savefig(f"{figdir}/Tu_V_kernel_PDF.png", dpi=150)
plt.close()


# 3). Create a Gaussian Kernel Density Estimator
kde_v = gaussian_kde(Tu_V_clean, bw_method=0.05)  # bw_method corresponds to seaborn's bw_adjust, controls the smoothing bandwidth

# 4). Define the range and number of points where the PDF will be evaluated, e.g., from -180° to 180° with 300 points
x_grid_v = np.linspace(-180, 180, 300)

# 5). Compute the PDF values at the specified points
pdf_values_v = kde_v(x_grid_v)



# Build an xarray Dataset
ds_out = xr.Dataset(
    {
        "Tu_deg": (["lat", "lon"], Tu_deg),
        "lon2d": (["lat", "lon"], lon2d),
        "lat2d": (["lat", "lon"], lat2d),
        "pdf_values_v": (["x_grid_v"], pdf_values_v),   # PDF values
        "x_grid_v": (["x_grid_v"], x_grid_v),           # Angle grid
    }
)

# save as NetCDF
ds_out.to_netcdf(f"{figdir}/Tu_difference.nc")

print("Saved output to figdir/Tu_difference.nc")
# Calculate the horizontal Turner Angle following Johnson et al. (2016) JPO
# Using instantaneous output or time-averages

# 1. Use plain fit to compute the gradients
# 2. Calculate the horizontal Turner Angle
# 3. Calculate the Kernel PDF distribution 


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
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_day1_3pm"

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# Set spatial indices
face = 1
k_surf = 0
i = slice(0,100,1) # Southern Ocean
j = slice(0,101,1) # Southern Ocean
# i = slice(1000,1200,1) # Tropics
# j = slice(2800,3001,1) # Tropics

# Grid spacings in m
dxF = ds1.dxF.isel(face=face,i=i,j=j)
dyF = ds1.dyF.isel(face=face,i=i,j=j)

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)

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
indices_15 = [i for i, t in enumerate(time_local) if t.hour == 15]  # 3pm local time
# print(indices_15)
# print(len(indices_15))

time_inst = indices_15[0]                



######### Load data #########

# Load surface T, S 
tt_surf = ds1.Theta.isel(time=time_inst,face=face,i=i,j=j,k=k_surf) # Potential temperature, shape (k, j, i)
ss_surf = ds1.Salt.isel(time=time_inst,face=face,i=i,j=j,k=k_surf)  # Practical salinity, shape (k, j, i)

# Compute the surface potential density using GSW (Gibbs Seawater Oceanography Toolkit), with surface reference pressure 
p_ref = 0                                            # Reference pressure 
SA_surf = gsw.conversions.SA_from_SP(ss_surf, p_ref, lon, lat) # Absolute salinity, shape (k, j, i)
CT_surf = gsw.conversions.CT_from_pt(SA_surf, tt_surf)              # Conservative temperature, shape (k, j, i)
rho_surf = gsw.density.rho(SA_surf, CT_surf, p_ref)                 # Potential density, shape (k, j, i)
alpha_surf = gsw.density.alpha(SA_surf, CT_surf, p_ref)             # Thermal expansion coefficient with respect to Conservative Temperature, shape (k, j, i)
beta_surf = gsw.density.beta(SA_surf, CT_surf, p_ref)               # Saline (i.e. haline) contraction coefficient of seawater at constant Conservative Temperature, shape (k, j, i)


#############################################################
######### 1. Use plain fit to compute the gradients #########
#############################################################

# 1). Convert surface temperature and salinity to NumPy arrays
tt_surf_np = tt_surf.values       # shape (j, i)
ss_surf_np = ss_surf.values       # shape (j, i)
ny, nx = tt_surf_np.shape

# 2). Initialize arrays to hold temperature and salinity gradients, filled with NaNs
dt_dx = np.full_like(tt_surf_np, np.nan)
dt_dy = np.full_like(tt_surf_np, np.nan)
ds_dx = np.full_like(ss_surf_np, np.nan)
ds_dy = np.full_like(ss_surf_np, np.nan)

# 3). Create relative coordinates for the window
x = np.tile([-1, 0, 1], 3)            # shape (9,)  x = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
y = np.repeat([-1, 0, 1], 3)          # shape (9,)  y = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
A = np.vstack([x, y, np.ones(9)]).T   # shape (9, 3)

# 4). Loop through interior points of the grid (excluding edges)
for j in range(1, ny - 1):
    for i in range(1, nx - 1):
        # 4.1 Extract local 3×3 temperature block （inclusive on the left, exclusive on the right）
        local_tt = tt_surf_np[j-1:j+2, i-1:i+2].flatten()
        local_ss = ss_surf_np[j-1:j+2, i-1:i+2].flatten()
        
        # 4.2 Skip window if any value is missing
        if np.any(np.isnan(local_tt)):
            continue
        if np.any(np.isnan(local_ss)):
            continue

        # 4.3 Fit the plane to temperature using least-squares
        coeffs_tt, _, _, _ = lstsq(A, local_tt, rcond=None)
        a_tt, b_tt, _ = coeffs_tt  # dT/dx = a_tt, dT/dy = b_tt
        coeffs_ss, _, _, _ = lstsq(A, local_ss, rcond=None)
        a_ss, b_ss, _ = coeffs_ss

        # 4.4 Store the gradients, normalized by physical grid spacing
        dt_dx[j, i] = a_tt / dxF[j, i]
        dt_dy[j, i] = b_tt / dyF[j, i]
        ds_dx[j, i] = a_ss / dxF[j, i]
        ds_dy[j, i] = b_ss / dyF[j, i]

# 5). Plot the horizontal gradients
vmax_t = np.nanmax(np.abs([dt_dx, dt_dy]))  # symmetric range for temperature gradients
vmax_s = np.nanmax(np.abs([ds_dx, ds_dy]))  # symmetric range for salinity gradients

fig, axs = plt.subplots(2,2,figsize=(15,10))

p1 = axs[0,0].pcolormesh(lon2d, lat2d, dt_dx, shading='auto', cmap='coolwarm', vmin=-vmax_t, vmax=vmax_t)
axs[0,0].set_title(r'Zonal Temp. Gradient ($\partial_x \theta$)')
axs[0,0].set_xlabel('Longitude')
axs[0,0].set_ylabel('Latitude')
cb1 = fig.colorbar(p1, ax=axs[0,0], label='(\u00B0C/m)')
cb1.formatter = ticker.ScalarFormatter(useMathText=True)
cb1.formatter.set_powerlimits((-2, 2)) 
cb1.update_ticks()

p2 = axs[0,1].pcolormesh(lon2d, lat2d, dt_dy, shading='auto', cmap='coolwarm', vmin=-vmax_t, vmax=vmax_t)
axs[0,1].set_title(r'Merid. Temp. Gradient ($\partial_y \theta$)')
axs[0,1].set_xlabel('Longitude')
axs[0,1].set_ylabel('Latitude')
cb2 = fig.colorbar(p2, ax=axs[0,1], label='(\u00B0C/m)')
cb2.formatter = ticker.ScalarFormatter(useMathText=True)
cb2.formatter.set_powerlimits((-2, 2))
cb2.update_ticks()

p3 = axs[1,0].pcolormesh(lon2d, lat2d, ds_dx, shading='auto', cmap='coolwarm', vmin=-vmax_s, vmax=vmax_s)
axs[1,0].set_title(r'Zonal Salinity Gradient ($\partial_x S$)')
axs[1,0].set_xlabel('Longitude')
axs[1,0].set_ylabel('Latitude')
cb3 = fig.colorbar(p3, ax=axs[1,0], label='(psu/m)')
cb3.formatter = ticker.ScalarFormatter(useMathText=True)
cb3.formatter.set_powerlimits((-2, 2))
cb3.update_ticks()

p4 = axs[1,1].pcolormesh(lon2d, lat2d, ds_dy, shading='auto', cmap='coolwarm', vmin=-vmax_s, vmax=vmax_s)
axs[1,1].set_title(r'Merid. Salinity Gradient ($\partial_y S$)')
axs[1,1].set_xlabel('Longitude')
axs[1,1].set_ylabel('Latitude')
cb4 = fig.colorbar(p4, ax=axs[1,1], label='(psu/m)')
cb4.formatter = ticker.ScalarFormatter(useMathText=True)
cb4.formatter.set_powerlimits((-2, 2))
cb4.update_ticks()

plt.tight_layout()
plt.savefig(f"{figdir}/surface_t_s_gradients.png", dpi=150)
plt.close()

############################################################
######### 2. Calculate the horizontal Turner Angle #########
############################################################

# 1). Calculate the across-isopycnal gradients
grad_rho_x = -alpha_surf * dt_dx + beta_surf * ds_dx
grad_rho_y = -alpha_surf * dt_dy + beta_surf * ds_dy

mag_linear = np.hypot(grad_rho_x, grad_rho_y) # magnitude of horizontal density gradient estimated based on the linear Equation of State

# 2). Define a unit vector (norm_x, norm_y) to represent the direction of the 2D horizontal density gradient
norm_x = grad_rho_x / mag_linear
norm_y = grad_rho_y / mag_linear

# 3). Across-isopycnal horizontal surface temperature and salinity gradient
dt_cross = dt_dx * norm_x + dt_dy * norm_y
ds_cross = ds_dx * norm_x + ds_dy * norm_y

# 4). Plot dt_cross and ds_cross
fig, axs = plt.subplots(1,2,figsize=(15,5))

p1 = axs[0].pcolormesh(lon2d, lat2d, dt_cross, shading='auto', cmap='coolwarm', vmin=-vmax_t, vmax=vmax_t)
axs[0].set_title(r'Cross-isop. Temp. Gradient ($\partial \theta$)')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
cb1 = fig.colorbar(p1, ax=axs[0], label='(\u00B0C/m)')
cb1.formatter = ticker.ScalarFormatter(useMathText=True)
cb1.formatter.set_powerlimits((-2, 2))  
cb1.update_ticks()

p2 = axs[1].pcolormesh(lon2d, lat2d, ds_cross, shading='auto', cmap='coolwarm', vmin=-vmax_s, vmax=vmax_s)
axs[1].set_title(r'Cross-isop. Salinity Gradient ($\partial S$)')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
cb2 = fig.colorbar(p2, ax=axs[1], label='(\u00B0C/m)')
cb2.formatter = ticker.ScalarFormatter(useMathText=True)
cb2.formatter.set_powerlimits((-2, 2))
cb2.update_ticks()

plt.tight_layout()
plt.savefig(f"{figdir}/surface_t_s_gradients_cross.png", dpi=150)
plt.close()

# 5). Horizontal Turner Angle
denominator= alpha_surf * dt_cross - beta_surf * ds_cross
numerator = alpha_surf * dt_cross + beta_surf * ds_cross

Tu_H_rad = np.arctan2(numerator, denominator)
Tu_H_deg = np.degrees(Tu_H_rad)

# print("Max Turner Angle (deg):", np.nanmax(Tu_H_deg.values))
# print("Min Turner Angle (deg):", np.nanmin(Tu_H_deg.values))

# 6). Plot a map of the Turner Angle (Tu)
plt.figure(figsize=(9, 6))
pcm = plt.pcolormesh(lon2d, lat2d, Tu_H_deg, cmap='twilight_shifted', shading='auto', vmin=-180, vmax=180)
plt.colorbar(pcm, label=r"$Tu_H$ (°)")
plt.title(r"Horizontal Turner Angle ($Tu_H$)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(f"{figdir}/hori_turner_angle_map.png", dpi=150)
plt.close()


############################################################
######### 3. Calculate the Kernel PDF distribution #########
############################################################

# 1). Flatten the Tu_H_deg array and remove NaN values
Tu_H_flat = Tu_H_deg.values.flatten()
Tu_H_clean = Tu_H_flat[~np.isnan(Tu_H_flat)]  # remove NaNs

# 2). Plot Kernel PDF
plt.figure(figsize=(8, 5))
sns.kdeplot(Tu_H_clean, bw_adjust=0.5, fill=True, color="darkblue", label=r'$Tu_H$ Kernel density estimation')
plt.xlabel(r'$Tu_H$ (°)')
plt.ylabel("Density")
plt.title("Kernel PDF of Horizontal Turner Angle")

plt.grid(True, which='major', linestyle=':')  # major grid lines
plt.minorticks_on()                           # enable minor ticks
plt.grid(True, which='minor', linestyle='--', alpha=0.15)  # minor grid lines (lighter and dashed)

# Highlight vertical lines at -90 and 90 degrees
plt.axvline(x=-90, color='red', linestyle='--', linewidth=2, label='-90°')
plt.axvline(x=90, color='red', linestyle='--', linewidth=2, label='90°')

plt.legend()
plt.tight_layout()
plt.savefig(f"{figdir}/Tu_H_kernel_PDF.png", dpi=150)
plt.close()


# 3). Create a Gaussian Kernel Density Estimator
kde_h = gaussian_kde(Tu_H_clean, bw_method=0.5)  # bw_method corresponds to seaborn's bw_adjust, controls the smoothing bandwidth

# 4). Define the range and number of points where the PDF will be evaluated, e.g., from -180° to 180° with 1000 points
x_grid_h = np.linspace(-180, 180, 1000)

# 5). Compute the PDF values at the specified points
pdf_values_h = kde_h(x_grid_h)

# 6). Test the sensitivity of KDE to smoothing bandwidth
for bw in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
    kde_h = gaussian_kde(Tu_H_clean, bw_method=bw)
    plt.plot(x_grid_h, kde_h(x_grid_h), label=f"bw={bw}")
plt.legend()
plt.savefig(f"{figdir}/Tu_H_kernel_PDF-test.png", dpi=150)
plt.close()


# Open the NETCDF file
ds_out = xr.open_dataset(f"{figdir}/Tu_difference.nc")

# Add to the dataset
ds_out["Tu_H_deg"] = (["lat", "lon"], Tu_H_deg.values)
ds_out["Tu_abs_diff"] = np.abs(ds_out["Tu_deg"] - Tu_H_deg.values)
ds_out["dt_dx"] = (["lat", "lon"], dt_dx)
ds_out["dt_dy"] = (["lat", "lon"], dt_dy)
ds_out["ds_dx"] = (["lat", "lon"], ds_dx)
ds_out["ds_dy"] = (["lat", "lon"], ds_dy)
ds_out["dt_cross"] = (["lat", "lon"], dt_cross.values)
ds_out["ds_cross"] = (["lat", "lon"], ds_cross.values)
ds_out["norm_x"] = (["lat", "lon"], norm_x.values)
ds_out["norm_y"] = (["lat", "lon"], norm_y.values)
ds_out["SA_surf"] = (["lat", "lon"], SA_surf.values)
ds_out["CT_surf"] = (["lat", "lon"], CT_surf.values)
ds_out["rho_surf"] = (["lat", "lon"], rho_surf.values)
ds_out["alpha_surf"] = (["lat", "lon"], alpha_surf.values)
ds_out["beta_surf"] = (["lat", "lon"], beta_surf.values)

# Add horizontal Turner angle PDF
ds_out["x_grid_h"] = (["x_grid_h"], x_grid_h)
ds_out["pdf_values_h"] = (["x_grid_h"], pdf_values_h)

# Save to file
ds_out.to_netcdf(f"{figdir}/Tu_difference.nc")

print("Saved variables to figdir/Tu_difference.nc")

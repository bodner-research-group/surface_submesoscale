# Calculate the horizontal Turner Angle following Johnson et al. (2016) JPO
# Using instantaneous output or time-averages

# Load packages
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr 
import dask 
import gsw  # (Gibbs Seawater Oceanography Toolkit) https://teos-10.github.io/GSW-Python/gsw.html
from numpy.linalg import lstsq

# Load the model
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Folder to store the figures
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_test1"

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# Set indices
face = 1
nday_avg = 7                 # 7-day average
time_avg = slice(0,24*nday_avg,1)  
time_inst = 0
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


# For each horizontal grid point, take the 8 grid points around it, and do a plain fit

# 1. Convert surface temperature and salinity to NumPy arrays
tt_surf_np = tt_surf.values       # shape (j, i)
ss_surf_np = ss_surf.values       # shape (j, i)
ny, nx = tt_surf_np.shape

# 2. Initialize arrays to hold temperature and salinity gradients, filled with NaNs
dt_dx = np.full_like(tt_surf_np, np.nan)
dt_dy = np.full_like(tt_surf_np, np.nan)
ds_dx = np.full_like(ss_surf_np, np.nan)
ds_dy = np.full_like(ss_surf_np, np.nan)

# 3. Create relative coordinates for the window
x = np.tile([-1, 0, 1], 3)            # shape (9,)  x = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
y = np.repeat([-1, 0, 1], 3)          # shape (9,)  y = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
A = np.vstack([x, y, np.ones(9)]).T   # shape (9, 3)

# 4. Loop through interior points of the grid (excluding edges)
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

# 5. Plot the horizontal gradients
fig, axs = plt.subplots(2,2,figsize=(15,10))

p1 = axs[0,0].pcolormesh(lon2d, lat2d, dt_dx, shading='auto', cmap='coolwarm')
axs[0,0].set_title('Zonal Gradient of Temperature (dT/dx)')
axs[0,0].set_xlabel('Longitude')
axs[0,0].set_ylabel('Latitude')
fig.colorbar(p1, ax=axs[0,0], label='(\u00B0C/m)')

p2 = axs[0,1].pcolormesh(lon2d, lat2d, dt_dy, shading='auto', cmap='coolwarm')
axs[0,1].set_title('Meridional Gradient of Temperature (dT/dy)')
axs[0,1].set_xlabel('Longitude')
axs[0,1].set_ylabel('Latitude')
fig.colorbar(p2, ax=axs[0,1], label='(\u00B0C/m)')

p3 = axs[1,0].pcolormesh(lon2d, lat2d, ds_dx, shading='auto', cmap='coolwarm')
axs[1,0].set_title('Zonal Gradient of Salinity (dS/dx)')
axs[1,0].set_xlabel('Longitude')
axs[1,0].set_ylabel('Latitude')
fig.colorbar(p1, ax=axs[1,0], label='(psu/m)')

p4 = axs[1,1].pcolormesh(lon2d, lat2d, ds_dy, shading='auto', cmap='coolwarm')
axs[1,1].set_title('Meridional Gradient of Salinity (dS/dy)')
axs[1,1].set_xlabel('Longitude')
axs[1,1].set_ylabel('Latitude')
fig.colorbar(p2, ax=axs[1,1], label='(psu/m)')

plt.tight_layout()
plt.savefig(f"{figdir}/surface_t_s_gradients.png", dpi=150)
plt.close()

# Calculate the horizontal Turner Angle

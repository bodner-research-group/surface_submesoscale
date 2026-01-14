"""
Compute Ekman buoyancy flux from LLC4320 hourly surf data.
Steps:
1. Compute Ekman buoyancy flux and save as NetCDF
2. Compute domain-averaged flux (exclude 2 boundary points), save as NetCDF
3. Plot hourly flux Jan 1–10 2011
"""

import os
import numpy as np
import xarray as xr
import gsw
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from xgcm import Grid
from dask.distributed import Client, LocalCluster

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=16, threads_per_worker=1, memory_limit="22GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Domain settings
# =====================
from set_constant import domain_name, face, i, j, start_hours, end_hours
time = slice(start_hours, end_hours, 1)
# time = slice(start_hours, start_hours+24, 1)

# =====================
# Load dataset
# =====================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# =====================
# Subset surface fields
# =====================
chunk_time = 24*7 
salt = ds1.Salt.isel(face=face, i=i, j=j, k=0, time=time).chunk({"time": chunk_time, "j": -1, "i": -1})
theta = ds1.Theta.isel(face=face, i=i, j=j, k=0, time=time).chunk({"time": chunk_time, "j": -1, "i": -1})

taux = ds1.oceTAUX.isel(face=face, i_g=i, j=j, time=time).chunk({"time": chunk_time, "j": -1, "i_g": -1})
tauy = ds1.oceTAUY.isel(face=face, i=i, j_g=j, time=time).chunk({"time": chunk_time, "j_g": -1, "i": -1})

lon = ds1.XC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
lat = ds1.YC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})

Coriolis = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)
rho0 = 1027.5
gravity = 9.81

# =====================
# xgcm grid
# =====================
ds_grid_face = ds1.isel(face=face, i=i, j=j, i_g=i, j_g=j, k=0, k_p1=0, k_u=0)

if "time" in ds_grid_face:
    ds_grid_face = ds_grid_face.isel(time=0, drop=True)

coords = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
}
metrics = {
    ("X",): ["dxC", "dxG"],
    ("Y",): ["dyC", "dyG"],
}
grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)


# =====================
# Compute potential density ρ(SA,CT)
# =====================
print("Computing SA...")
SA = xr.apply_ufunc(
    gsw.SA_from_SP, salt, 0, lon, lat,
    input_core_dims=[["j","i"], [], ["j","i"], ["j","i"]],
    output_core_dims=[["j","i"]],
    vectorize=True, dask="parallelized", output_dtypes=[np.float32],
)
SA = SA.compute()  # Ensure SA is computed


print("Computing CT...")
CT = xr.apply_ufunc(
    gsw.CT_from_pt, SA, theta,
    input_core_dims=[["j","i"], ["j","i"]],
    output_core_dims=[["j","i"]],
    vectorize=True, dask="parallelized", output_dtypes=[np.float32],
)
CT = CT.compute()  # Ensure CT is computed


print("Computing density (rho)...")
rho = xr.apply_ufunc(
    gsw.rho, SA, CT, 0,
    input_core_dims=[["j","i"], ["j","i"], []],
    output_core_dims=[["j","i"]],
    vectorize=True, dask="parallelized", output_dtypes=[np.float32],
)
rho = rho.compute()  # Ensure rho is computed

del SA, CT
# rho = rho.astype(np.float32)

# =====================
# Surface buoyancy
# =====================
buoy = -gravity * (rho - rho0) / rho0

buoy = buoy.compute()
del rho0

# =====================
# Compute gradients ∂b/∂x, ∂b/∂y
# =====================
def grad_center(var): 
    dx = grid.derivative(var, axis="X") 
    dy = grid.derivative(var, axis="Y") 
    dx = grid.interp(dx, axis="X", to="center") 
    dy = grid.interp(dy, axis="Y", to="center") 
    return dx, dy


print("Computing gradients of buoyancy...")
dbdx, dbdy = grad_center(buoy)

dbdx = dbdx.compute()  # Ensure dbdx is computed
dbdy = dbdy.compute()  # Ensure dbdy is computed

dbdx = dbdx.chunk({"j": -1, "i": -1})
dbdy = dbdy.chunk({"j": -1, "i": -1})

del buoy

# =====================
# Compute wind stress (taux_c, tauy_c) 
# =====================
print("Interpolating wind stress to C-center (taux_c)...")
taux_c = grid.interp(taux, axis="X", to="center")
taux_c = taux_c.compute()  # Ensure taux_c is computed

print("Interpolating wind stress to C-center (tauy_c)...")
tauy_c = grid.interp(tauy, axis="Y", to="center")
tauy_c = tauy_c.compute()  # Ensure tauy_c is computed

# =====================
# Ekman buoyancy flux B_Ek
# =====================
B_Ek = (tauy_c * dbdx - taux_c * dbdy) / (rho0 * Coriolis)
B_Ek = B_Ek.rename("B_Ek")
B_Ek.attrs["units"] = "m^2/s^3"
B_Ek.attrs["long_name"] = "Ekman Buoyancy Flux"
B_Ek = B_Ek.compute()  # Ensure B_Ek is computed

# =====================
# Save Full Field as NetCDF
# =====================
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(out_dir, exist_ok=True)

print("Saving full Ekman buoyancy flux...")
fout_field = f"{out_dir}/Ekman_buoyancy_flux_hourly.nc"
B_Ek.to_netcdf(fout_field)

# =====================
# Compute domain-mean B_Ek (exclude 2 boundary points)
# =====================
print("Computing domain-averaged Ekman buoyancy flux...")
Bsub = B_Ek.isel(i=slice(2, -2), j=slice(2, -2))
Bmean = Bsub.mean(dim=("i", "j")).rename("B_Ek_mean")

fout_mean = f"{out_dir}/Ekman_buoyancy_flux_hourly_domain_mean.nc"
Bmean.to_netcdf(fout_mean)

# =====================
# Plot time series Jan 1–10, 2011
# =====================
print("Plotting hourly Ekman buoyancy flux Jan 1–10, 2011...")

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(figdir, exist_ok=True)

# Convert model time to datetime
t = xr.decode_cf(Bmean).time

# Select time window
B10 = Bmean.sel(time=slice("2011-01-01", "2011-01-10"))

plt.figure(figsize=(12, 5))
plt.plot(B10.time, B10, lw=1.3)
plt.grid(True)
plt.title(f"Ekman Buoyancy Flux (Domain Mean)\n{domain_name} | Jan 1–10, 2011")
plt.ylabel("B_Ek (m²/s³)")
plt.xlabel("Time (UTC)")
plt.tight_layout()

fig_out = f"{out_dir}/B_Ek_timeseries_Jan1_10_2011.png"
plt.savefig(fig_out, dpi=150)
plt.close()

print("DONE.")









# =====================
# 1. Plot Timeseries (Domain-averaged EBF)
# =====================
print("Plotting domain-averaged Ekman buoyancy flux with weekly rolling mean...")

# Convert model time to datetime for plotting (if it's not already)
t = xr.decode_cf(Bmean).time

# Plotting the domain-averaged Ekman buoyancy flux
plt.figure(figsize=(12, 6))

# Plot the domain-mean EBF (original)
plt.plot(t, Bmean, lw=1.5, label="Domain-averaged EBF", color='blue')

# Calculate a 7-day (weekly) rolling mean of the domain-averaged EBF
Bmean_rolling = Bmean.rolling(time=7, center=True).mean()

# Plot the weekly rolling mean
plt.plot(t, Bmean_rolling, lw=2, label="7-day Rolling Mean", color='red')

# Add labels and title
plt.title(f"Domain Averaged Ekman Buoyancy Flux with 7-Day Rolling Mean\n{domain_name}", fontsize=16)
plt.xlabel("Time (UTC)", fontsize=14)
plt.ylabel("B_Ek (m²/s³)", fontsize=14)
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()

# Save the figure
fig_out = f"{out_dir}/B_Ek_timeseries_with_rolling_mean.png"
plt.savefig(fig_out, dpi=150)
plt.close()

print(f"Figure saved to: {fig_out}")





import matplotlib.pyplot as plt
import numpy as np

# =====================
# 2. Plot Maps of EBF from Jan 1 to Jan 10
# =====================
print("Plotting maps of Ekman buoyancy flux from Jan 1–10, 2011...")

# Select data from Jan 1 to Jan 10, 2011
B10 = B_Ek.sel(time=slice("2011-01-01", "2011-01-10"))

# Set up figure
fig, ax = plt.subplots(figsize=(12, 6))

# Define the number of time steps (hours)
times = B10.time.values

# Create a plot for each time step
for i, time_step in enumerate(times):
    # Select the Ekman buoyancy flux for the current time step
    B_current = B10.sel(time=time_step)
    
    # Create a contour plot
    im = ax.contourf(lon, lat, B_current, levels=np.linspace(-np.max(np.abs(B_current)), np.max(np.abs(B_current)), 21),
                     cmap='RdBu_r', extend='both')

    # Title with the current time
    ax.set_title(f"Ekman Buoyancy Flux at {np.datetime_as_string(time_step)}", fontsize=14)
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("Ekman Buoyancy Flux (m²/s³)", fontsize=12)

    # Set plot limits for latitude and longitude
    ax.set_xlim([lon.min(), lon.max()])
    ax.set_ylim([lat.min(), lat.max()])
    
    # Add grid
    ax.grid(True)

    # Save figure for each time step
    fig_out_map = f"{figdir}/EBF_map_{np.datetime_as_string(time_step)}.png"
    plt.savefig(fig_out_map, dpi=150)
    plt.clf()  # Clear the plot to avoid overlap

    print(f"Saved map for {np.datetime_as_string(time_step)} to {fig_out_map}")

plt.close()

print("DONE.")
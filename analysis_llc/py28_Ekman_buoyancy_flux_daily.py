##### Compute the contribution of Ekman buoyancy flux to the change in mixed layer depth
##### Following Thomas (2005), Thomas & Lee (2005), Thomas & Ferrari (2008), Thompson et al. (2016), Johnson et al. (2020b), etc.
#####
##### The change in mixed layer stratification (\partial_t b_z)^{Ek} ~ (\tau^x\partial_y b - \tau^y\partial_x b)/rho0/f/h^2, where h is a depth. 
##### h can be the mixed layer depth, the convective layer depth, the Ekman layer depth, or the KPP boundary layer depth??
#####

import os
import numpy as np
import xarray as xr
from glob import glob
from xgcm import Grid

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
from tqdm.auto import tqdm

# ========== Domain ==========
from set_constant import domain_name, face, i, j
# domain_name = "icelandic_basin"
# face = 2
# i = slice(527, 1007)   # icelandic_basin -- larger domain
# j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ==============================================================
# Constants
# ==============================================================
g = 9.81
rho0 = 1027.5

# ==============================================================
# Paths
# ==============================================================
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
b_gradient_file  = os.path.join(base_dir, "ML_buoyancy_gradients_daily.nc")
wind_stress_file = os.path.join(base_dir, "windstress_center_daily_avg.nc")
out_dir  = base_dir

taux = xr.open_dataset(wind_stress_file).taux_center
tauy = xr.open_dataset(wind_stress_file).tauy_center
dbdx = xr.open_dataset(b_gradient_file).dbdx
dbdy = xr.open_dataset(b_gradient_file).dbdy
Hml = xr.open_dataset(b_gradient_file).Hml


# ==============================================================
# Load grid + wind stress (Dask lazy)
# ==============================================================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

lon  = ds1.XC.isel(face=face, i=i, j=j)
lat  = ds1.YC.isel(face=face, i=i, j=j)
depth = ds1.Z
drF   = ds1.drF
drF3d, _, _ = xr.broadcast(drF, lon, lat)

Coriolis = 4*np.pi/86164*np.sin(lat*np.pi/180)
f0 = Coriolis.mean(dim=("i","j")).values     ### Coriolis parameter averaged over this domain

# Ekman buoyancy flux 
B_Ek = (taux * dbdy - tauy * dbdx) / (rho0 * Coriolis)   # (time,i,j) at center grid

B_Ek_d_Hml2 = B_Ek/(Hml**2)




# ============================
# Paths
# ============================
# base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
# B_Ek_file = os.path.join(base_dir, "B_Ek.nc")       # <-- you create this earlier
out_ts = os.path.join(base_dir, "B_Ek_timeseries.nc")

fig_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(fig_dir, exist_ok=True)
out_mp4 = os.path.join(fig_dir, "B_Ek_animation.mp4")

# ============================
# Load data
# ============================
# B_Ek = xr.open_dataset(B_Ek_file).B_Ek    # shape (time,j,i)
# lon = B_Ek.lon
# lat = B_Ek.lat
time = B_Ek.time

# # ============================
# # 1. Plot daily maps → frames
# # ============================

# print("\nCreating daily maps...")

# for n in tqdm(range(B_Ek.time.size)):
#     day = B_Ek.isel(time=n)
#     date_str = np.datetime_as_string(day.time.values, unit='D')

#     fig = plt.figure(figsize=(10, 7))
#     ax = plt.axes(projection=ccrs.PlateCarree())

#     pcm = ax.pcolormesh(
#         lon, lat, day,
#         cmap="RdBu_r",
#         vmin=-5e-8, vmax=5e-8,   # adjust ranges
#         shading='auto',
#         transform=ccrs.PlateCarree(),
#     )

#     ax.coastlines(color='k', linewidth=0.8)
#     ax.set_title(f"Ekman Buoyancy Flux B_Ek — {date_str}", fontsize=14)
#     cbar = plt.colorbar(pcm, ax=ax, label="m²/s³")

#     frame_file = os.path.join(fig_dir, f"B_Ek_{date_str}.png")
#     plt.savefig(frame_file, dpi=150)
#     plt.close()

# print("Frames saved.")

# # ============================
# # Create MP4 animation
# # ============================

# print("\nCreating MP4 animation...")

# frame_list = sorted([os.path.join(fig_dir, f) for f in os.listdir(fig_dir) if f.endswith(".png")])

# fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection=ccrs.PlateCarree())

# def animate(k):
#     img = plt.imread(frame_list[k])
#     ax.clear()
#     ax.imshow(img, extent=[-10,10,-10,10])  # dummy extent, replaced by full frame
#     return []

# ani = animation.FuncAnimation(fig, animate, frames=len(frame_list), interval=120)
# ani.save(out_mp4, writer='ffmpeg', dpi=150)

# plt.close()
# print(f"MP4 saved to: {out_mp4}")

# ============================
# 2. Domain-averaged timeseries
# ============================
print("\nComputing domain averaged B_Ek...")

# # simple spatial mean
# B_Ek_mean = B_Ek.mean(dim=("j", "i"))

# ============================
# 1. Remove boundary points
# ============================
B_inner  = B_Ek.isel(j=slice(2, -2), i=slice(2, -2))
Hml_inner = Hml.isel(j=slice(2, -2), i=slice(2, -2))

# ============================
# 2. Mask NaNs & zero values
# ============================
B_masked = B_inner.where(~np.isnan(B_inner))
B_masked = B_masked.where(B_masked != 0)

Hml_masked = Hml_inner.where(~np.isnan(Hml_inner))
Hml_masked = Hml_masked.where(Hml_masked != 0)

# ============================
# 3. Domain-mean B_Ek
# ============================
B_Ek_mean = B_masked.mean(dim=("j", "i"), skipna=True)

# ============================
# 4. Compute B_Ek_d_Hml2
# ============================
B_Ek_d_Hml2 = B_masked / (Hml_masked ** 2)

# Mask infinities just in case
B_Ek_d_Hml2 = B_Ek_d_Hml2.where(np.isfinite(B_Ek_d_Hml2))

# Domain mean
B_Ek_d_Hml2_mean = B_Ek_d_Hml2.mean(dim=("j", "i"), skipna=True)


ds_ts = xr.Dataset({
    "B_Ek_mean": B_Ek_mean,
    "B_Ek_d_Hml2_mean": B_Ek_d_Hml2_mean,
})
ds_ts.to_netcdf(out_ts)
print(f"Saved timeseries to: {out_ts}")

# ============================
# 3. Plot timeseries
# ============================

print("\nPlotting timeseries...")
out_fig = os.path.join(fig_dir, "B_Ek_timeseries.png")
plt.figure(figsize=(12,5))
plt.plot(B_Ek_mean.time, B_Ek_mean, '-k', lw=1.2)
plt.axhline(0, color='gray', ls='--')
plt.ylabel("B_Ek (m²/s³)")
plt.xlabel("Time")
plt.title("Domain-averaged Ekman Buoyancy Flux")

plt.tight_layout()
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"Timeseries figure saved to: {out_fig}")
print("\nDONE.")

out_fig = os.path.join(fig_dir, "B_Ek_d_Hml2_mean_timeseries.png")
plt.figure(figsize=(12,5))
plt.plot(B_Ek_d_Hml2_mean.time, B_Ek_d_Hml2_mean, '-k', lw=1.2)
plt.axhline(0, color='gray', ls='--')
plt.ylabel("B_Ek (m²/s³)")
plt.xlabel("Time")
plt.title("Ekman Buoyancy Flux/Hml^2")

plt.tight_layout()
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"Timeseries figure saved to: {out_fig}")
print("\nDONE.")
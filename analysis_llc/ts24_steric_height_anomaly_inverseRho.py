#### Compute the steric height anomaly, following ECCO V4 Python Tutorial created by Andrew Delman:
#### https://ecco-v4-python-tutorial.readthedocs.io/Steric_height.html


# ===== Imports =====
import os
import numpy as np
import xarray as xr
import gsw
from datetime import timedelta
# from dask.distributed import Client, LocalCluster
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from xgcm import Grid

# from set_constant import domain_name, face, i, j

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain


# # =====================
# # Setup Dask cluster
# # =====================
# cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
# client = Client(cluster)
# print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly"
os.makedirs(output_dir, exist_ok=True)

# ========== Open LLC4320 dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# ========== Extract subregion ==========
lon = ds1['XC'].isel(face=face, i=i, j=j)
lat = ds1['YC'].isel(face=face, i=i, j=j)
depth = ds1.Z                    # 1D vertical coordinate
drF = ds1.drF  # vertical grid spacing, 1D
drF3d, _, _ = xr.broadcast(drF, lon, lat)

# ========== Open 7-day rolling mean of T, S, potential density, alpha, and beta ==========
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_insitu_hydrostatic_pressure_7d_rolling_mean"

# ========== Define constants ==========
g = 9.81
rhoConst = 1029.
p_atm = 101325./1e4   # atmospheric pressure at sea surface, in dbar

# ========== Load SSH ==========
# print("Loading daily averaged Eta...")
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds_eta = xr.open_mfdataset(eta_path, combine='by_coords')

# Daily averaged SSH
Eta_daily = ds_eta["Eta"] # Align datasets and select face/i/j region

# Compute 7-day rolling mean of SSH
Eta = Eta_daily.rolling(time=7, center=True).mean() 

# ===================================================
# ========== Compute steric height anomaly ==========
# ===================================================

eta = Eta.isel(time=170)
eta.time.values

# ========== Load in-situ densiy, SA, CT, and hydrostatic pressure ==========
input_file = os.path.join(input_dir, "rho_insitu_pres_hydro_20120419.nc")
date_tag = os.path.basename(input_file).split("_")[-1].replace(".nc", "")
ds = xr.open_dataset(input_file, chunks={"k": -1, "j": 50, "i": 50})

rho_insitu = ds.rho_insitu
pres_hydro = ds.pres_hydro
SA = ds.SA
CT = ds.CT

# ========== Specific volume anomaly ==========
# compute standard specific volume and anomalies
S_Ar = 35.16504    # absolute salinity standard for spec. vol., notated as SSO in GSW documentation
T_Cr = 0.          # conservative temperature standard
specvol_standard = gsw.density.specvol(S_Ar,T_Cr,pres_hydro.values)
specvol_constant = 1/rhoConst

specvol_mean = (1 / rho_insitu).mean(dim=["i", "j"])

# specvol_ref = specvol_standard
# specvol_ref = specvol_constant
specvol_ref = specvol_mean

specvol_anom = 1/rho_insitu - specvol_ref

# ========== Steric height anomaly ==========
# pressure reference level to compute steric height
# (in units of dbar, minus 10.1325 dbar atmospheric pressure)
p_top_sea_dbar = 0.
p_top = (p_top_sea_dbar) + p_atm ### dbar
p_r_sea_dbar = 1000.
p_r = (p_r_sea_dbar) + p_atm     ### dbar

# compute pressure at z = 0 (not exactly the ocean surface)
# press_z0 = p_atm + g*rho_insitu.isel(k=0)*eta /1e4 #### When computing steric height, we shouldn't use the information of SSH (eta)!
# press_z0 = g*rho_insitu.isel(k=0)*eta /1e4
press_z0 = p_atm+0*rho_insitu.isel(k=0)  # use atmospheric pressure as an estimation of the pressure at z=0

# integrate hydrostatic balance downward to get pressure at bottom of grid cells
press_ku = press_z0 + (rho_insitu*g*ds1.drF).cumsum("k")/1e4
press_ku = press_ku.assign_coords(Z=("k", ds1.Zu.values))

press_z0 = press_z0.expand_dims(k=[0])
press_z0 = press_z0.assign_coords(Z=("k", [ds1.Zl.isel(k_l=0).values]))

# create array with pressure at top of grid cells
press_kl = xr.concat([press_z0,press_ku.isel(k=np.arange(len(ds1.k) - 1))],dim="k")
press_kl = press_kl.assign_coords(k=ds1.k.values)

# compute dp for this integration
dp_integrate =  np.fmax(press_kl,p_top*np.ones(press_kl.shape)) - \
                np.fmin(press_ku,p_r*np.ones(press_ku.shape))        ### dbar 

# # allow integration above z=0 if p_top is less than p at z=0
# p_top_above_z0_mask = (p_top - press_kl.isel(k=0).values < 0)
# dp_integrate.isel(k=0).values[p_top_above_z0_mask] = \
#                                 (p_top - press_ku[:,0,:,:,:].values)[p_top_above_z0_mask]
dp_integrate.values[dp_integrate.values > 0] = 0

# Integrate specific volume anomaly over depth
k_range = slice(0,52)
steric_height_anom = (-(specvol_anom.isel(k=k_range)/g)*dp_integrate.isel(k=k_range)*1e4).sum("k") ### in meters

# ========== Compare steric height with sea surface height ==========
ssh_diff = eta - steric_height_anom

# eta_minus_mean = eta
eta_minus_mean = eta-eta.mean(dim=["i", "j"])
# steric_height_anom_minus_mean = steric_height_anom
steric_height_anom_minus_mean =  steric_height_anom-steric_height_anom.mean(dim=["i", "j"])



# ========= Load grid data =========
# print("Loading grid...")
ds_grid_face = ds1.isel(face=face,i=i, j=j,i_g=i, j_g=j,k=0,k_p1=0,k_u=0)

# Drop time dimension if exists
if 'time' in ds_grid_face.dims:
    ds_grid_face = ds_grid_face.isel(time=0, drop=True)  # or .squeeze('time')

# ========= Setup xgcm grid =========
coords = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
}
metrics = {
    ("X",): ["dxC", "dxG"],
    ("Y",): ["dyC", "dyG"],
}
grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)

# ========= Compute derivatives and Laplacian =========
def compute_grad_laplace(var, grid):
    
    # First derivatives on grid edges
    var_x = grid.derivative(var, axis="X")  # ∂var/∂x
    var_y = grid.derivative(var, axis="Y")  # ∂var/∂y
    # Interpolate first derivatives back to cell centers for gradient magnitude
    var_x_c = grid.interp(var_x, axis="X", to="center")
    var_y_c = grid.interp(var_y, axis="Y", to="center")
    grad_mag = np.sqrt(var_x_c**2 + var_y_c**2)
    grad_mag = grad_mag.assign_coords(time=var.time)

    # Second derivatives for Laplacian
    var_xx = grid.derivative(var_x_c, axis="X") # ∂²var/∂x²
    var_yy = grid.derivative(var_y_c, axis="Y") # ∂²var/∂y²
    # Interpolate second derivatives to cell centers
    var_xx_c = grid.interp(var_xx, axis="X", to="center")
    var_yy_c = grid.interp(var_yy, axis="Y", to="center")
    laplace = var_xx_c + var_yy_c
    laplace = laplace.assign_coords(time=var.time)

    return grad_mag, laplace

# Compute for SSH
eta_grad_mag, eta_laplace = compute_grad_laplace(eta_minus_mean, grid)

# Compute for steric height anomaly
steric_grad_mag, steric_laplace = compute_grad_laplace(steric_height_anom_minus_mean, grid)



### Path to the folder where figures will be saved 
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/steric_height/"
os.makedirs(figdir, exist_ok=True)

i_min = i.start
i_max = i.stop
j_min = j.start
j_max = j.stop

# Plot function
def plot_map(var, lon, lat, title, cmap, vmin=None, vmax=None, filename='output.png'):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(lon, lat, var, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.title(title + f"\n(face {face}, i={i_min}-{i_max}, j={j_min}-{j_max})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved plot: {filename}")

# Plot SSH
plot_map(
    var=eta_minus_mean,
    lon=lon,
    lat=lat,
    title="SSH",
    cmap="coolwarm",
    vmin=-0.2,
    vmax=0.2,
    filename=f"{figdir}SSH.png"
)

# Plot steric height anomaly
plot_map(
    var=steric_height_anom_minus_mean,
    lon=lon,
    lat=lat,
    title="Steric height anomaly",
    cmap="coolwarm",
    vmin=-0.2,
    vmax=0.2,
    filename=f"{figdir}steric_height_anom.png"
)



# =====================================================
# Plot results
# =====================================================

plot_map(
    var=eta_grad_mag,
    lon=lon,
    lat=lat,
    title="|∇ SSH| (m/m)",
    vmin=0,
    vmax=5e-6,
    cmap="viridis",
    filename=f"{figdir}SSH_gradmag_Apr_old.png"
)

plot_map(
    var=eta_laplace,
    lon=lon,
    lat=lat,
    title="∇² SSH (1/m²)",
    vmin=-1e-9,
    vmax=1e-9,
    cmap="coolwarm",
    filename=f"{figdir}SSH_laplace_Apr_old.png"
)

plot_map(
    var=steric_grad_mag,
    lon=lon,
    lat=lat,
    title="|∇ Steric Height Anomaly| (m/m)",
    vmin=0,
    vmax=5e-6,
    cmap="viridis",
    filename=f"{figdir}steric_gradmag_Apr_inverse.png"
)

plot_map(
    var=steric_laplace,
    lon=lon,
    lat=lat,
    title="∇² Steric Height Anomaly (1/m²)",
    vmin=-1e-9,
    vmax=1e-9,
    cmap="coolwarm",
    filename=f"{figdir}steric_laplace_Apr_inverse.png"
)

print("✅ Gradient magnitude and Laplacian maps saved.")











# # ===============================================================
# # ========== Thermosteric and halosteric contributions ==========
# # ===============================================================

# # Thermosteric: keep salinity constant at mean value
# salt_mean = salt.mean(dim='k')
# SA_mean = gsw.SA_from_SP(salt_mean, pres_hydro, lon, lat)
# CT_temp = gsw.CT_from_pt(SA_mean, theta)
# rho_temp = gsw.rho(SA_mean, CT_temp, pres_hydro)
# specvol_anom_temp = (1 / rho_temp) - (1 / rhoConst)
# thermosteric_height = (specvol_anom_temp * drF3d).sum(dim='k')

# # Halosteric: keep temperature constant at mean value
# theta_mean = theta.mean(dim='k')
# CT_salt = gsw.CT_from_pt(SA, theta_mean)
# rho_salt = gsw.rho(SA, CT_salt, pres_hydro)
# specvol_anom_salt = (1 / rho_salt) - (1 / rhoConst)
# halosteric_height = (specvol_anom_salt * drF3d).sum(dim='k')

# # ========== Save output ==========
# out_ds = xr.Dataset(
#     {
#         "steric_height_anomaly": steric_height_anom,
#         "thermosteric_height": thermosteric_height,
#         "halosteric_height": halosteric_height,
#     },
#     coords={
#         "i": ds.i,
#         "j": ds.j,
#         "face": face,
#         "XC": lon,
#         "YC": lat,
#     },
#     attrs={
#         "description": "Steric height anomaly and thermosteric/halosteric contributions",
#         "rhoConst": rhoConst,
#         "date": date_tag,
#     }
# )

# output_file = os.path.join(output_dir, f"steric_height_anomaly_{date_tag}.nc")
# out_ds.to_netcdf(output_file)

# print(f"💡 Saved steric height anomaly data to {output_file}")



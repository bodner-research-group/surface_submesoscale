

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

from set_constant import domain_name, face, i, j

# # ========== Domain ==========
# domain_name = "icelandic_basin"
# face = 2
# i = slice(527, 1007)   # icelandic_basin -- larger domain
# j = slice(2960, 3441)  # icelandic_basin -- larger domain

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

# ========== Open T, S, potential density, alpha, and beta ==========
# input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_insitu_hydrostatic_pressure_7d_rolling_mean"
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_insitu_hydrostatic_pressure_daily"

# ========== Define constants ==========
g = 9.81
rho0 = 1027.5
p_atm = 101325./1e4   # atmospheric pressure at sea surface, in dbar

# ========== Load SSH ==========
# print("Loading daily averaged Eta...")
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds_eta = xr.open_mfdataset(eta_path, combine='by_coords')

# Daily averaged SSH
Eta_daily = ds_eta["Eta"] # Align datasets and select face/i/j region

# Compute 7-day rolling mean of SSH
# Eta = Eta_daily.rolling(time=7, center=True).mean() 
Eta = Eta_daily

eta = Eta.isel(time=170)
eta.time.values

fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_submesoscale_30kmCutoff.nc" 
eta_submeso = xr.open_dataset(fname).SSH_submesoscale.isel(time=170)
eta_submeso.time.values

eta_minus_mean = eta-eta.mean(dim=["i", "j"])
eta_submeso_minus_mean =  eta_submeso-eta_submeso.mean(dim=["i", "j"])


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
eta_submeso_grad_mag, eta_submeso_laplace = compute_grad_laplace(eta_submeso_minus_mean, grid)


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
    filename=f"{figdir}SSH_Apr.png"
)

# Plot steric height anomaly
plot_map(
    var=eta_submeso_minus_mean,
    lon=lon,
    lat=lat,
    title="Steric height anomaly",
    cmap="coolwarm",
    vmin=-0.2,
    vmax=0.2,
    filename=f"{figdir}Submeso_height_anom_Apr.png"
)



# =====================================================
# Plot results
# =====================================================

plot_map(
    var=eta_submeso_grad_mag,
    lon=lon,
    lat=lat,
    title="|∇ Submesoscale SSH Anomaly| (m/m)",
    vmin=0,
    vmax=5e-6,
    cmap="viridis",
    filename=f"{figdir}submeso_gradmag_Apr.png"
)

plot_map(
    var=eta_submeso_laplace,
    lon=lon,
    lat=lat,
    title="∇² Submesoscale SSH Anomaly (1/m²)",
    vmin=-1e-9,
    vmax=1e-9,
    cmap="coolwarm",
    filename=f"{figdir}submeso_laplace_Apr.png"
)

print("✅ Gradient magnitude and Laplacian maps saved.")


















##############################################################################
##############################################################################
##############################################################################
##############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and plot steric height anomaly, its gradient magnitude, and Laplacian
for multiple SSH datasets following Eq.(1) of Jinbo Wang et al. (2025)
"""

# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xgcm import Grid

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # Icelandic Basin domain
j = slice(2960, 3441)

# ========== Directories ==========
ssh_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/steric_height"
os.makedirs(figdir, exist_ok=True)

# ========== Open grid information ==========
print("🔹 Loading LLC4320 grid subset...")
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lon = ds1["XC"].isel(face=face, i=i, j=j)
lat = ds1["YC"].isel(face=face, i=i, j=j)

# Extract grid face info
ds_grid_face = ds1.isel(face=face, i=i, j=j, i_g=i, j_g=j, k=0, k_p1=0, k_u=0)
if "time" in ds_grid_face.dims:
    ds_grid_face = ds_grid_face.isel(time=0, drop=True)

coords = {"X": {"center": "i", "left": "i_g"}, "Y": {"center": "j", "left": "j_g"}}
metrics = {("X",): ["dxC", "dxG"], ("Y",): ["dyC", "dyG"]}
grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)

# ========== Helper functions ==========
def compute_grad_laplace(var, grid):
    """Compute gradient magnitude and Laplacian of a variable using xgcm."""
    var_x = grid.derivative(var, axis="X")
    var_y = grid.derivative(var, axis="Y")

    var_x_c = grid.interp(var_x, axis="X", to="center")
    var_y_c = grid.interp(var_y, axis="Y", to="center")
    grad_mag = np.sqrt(var_x_c**2 + var_y_c**2)
    grad_mag = grad_mag.assign_coords(time=var.time)

    var_xx = grid.derivative(var_x_c, axis="X")
    var_yy = grid.derivative(var_y_c, axis="Y")
    var_xx_c = grid.interp(var_xx, axis="X", to="center")
    var_yy_c = grid.interp(var_yy, axis="Y", to="center")
    laplace = var_xx_c + var_yy_c
    laplace = laplace.assign_coords(time=var.time)

    return grad_mag, laplace


def plot_map(var, lon, lat, title, cmap, vmin=None, vmax=None, filename="output.png"):
    """Simple plotting function for maps."""
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(lon, lat, var, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved plot: {filename}")


# ========== Files to process ==========
ssh_files = [
    # "SSH_Gaussian_meso_30kmCutoff_1_12deg.nc",
    # "SSH_Gaussian_meso_30kmCutoff.nc",
    # "SSH_Gaussian_submeso_30kmCutoff.nc",
    # "SSH_GCMFilters_meso_30kmCutoff_1_12deg.nc",
    # "SSH_GCMFilters_meso_30kmCutoff.nc",
    # "SSH_GCMFilters_submeso_30kmCutoff.nc",
    "SSH_RollingMean_submeso_10kmCutoff.nc",
    "SSH_RollingMean_submeso_20kmCutoff.nc",
    "SSH_RollingMean_submeso_30kmCutoff.nc",
]

# ========== Loop over each SSH file ==========
for fname in ssh_files:
    fpath = os.path.join(ssh_dir, fname)
    print(f"\n🔹 Processing: {fname}")

    # Load SSH dataset
    ds = xr.open_dataset(fpath)
    ssh_varname = [v for v in ds.data_vars][0]  # assume only one SSH variable
    ssh = ds[ssh_varname].isel(time=170)  # same time index as example

    # Remove domain mean
    ssh_anom = ssh - ssh.mean(dim=["i", "j"])

    # Compute gradient and Laplacian
    grad_mag, laplace = compute_grad_laplace(ssh_anom, grid)

    # Save plots
    base = fname.replace(".nc", "")
    plot_map(
        var=ssh_anom,
        lon=lon,
        lat=lat,
        title=f"{base} anomaly (m)",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=0.2,
        filename=f"{figdir}/{base}_anom.png",
    )

    plot_map(
        var=grad_mag,
        lon=lon,
        lat=lat,
        title=f"|∇ {base}| (m/m)",
        cmap="viridis",
        vmin=0,
        vmax=5e-6,
        filename=f"{figdir}/{base}_gradmag.png",
    )

    plot_map(
        var=laplace,
        lon=lon,
        lat=lat,
        title=f"∇² {base} (1/m²)",
        cmap="coolwarm",
        vmin=-1e-9,
        vmax=1e-9,
        filename=f"{figdir}/{base}_laplace.png",
    )

print("\n✅ All steric height plots computed and saved.")

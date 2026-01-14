###### This takes too long to run -- 17 minutes for a 10x10 grid domain 

import os
import numpy as np
import xarray as xr
import gsw
from xgcm import Grid
from dask.distributed import Client, LocalCluster

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB",
    processes=True
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Domain settings
# =====================
domain_name = "test_100grids"
face = 2
i = slice(527, 537)
j = slice(2960, 2970)

# from set_constant import domain_name, face, i, j

# domain_name = "icelandic_basin"
# face = 2
# i = slice(527, 1007)   # icelandic_basin -- larger domain
# j = slice(2960, 3441)  # icelandic_basin -- larger domain


# =====================
# Load dataset
# =====================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# Select subdomain
salt = ds1.Salt.isel(face=face, i=i, j=j).chunk({"time": 1, "k": -1, "j": -1, "i": -1})
theta = ds1.Theta.isel(face=face, i=i, j=j).chunk({"time": 1, "k": -1, "j": -1, "i": -1})
lon = ds1.XC.isel(face=face, i=i, j=j)
lat = ds1.YC.isel(face=face, i=i, j=j)
depth = ds1.Z  # 1D k
drF = ds1.drF  # 1D k

# Broadcast depth and drF to 3D (k,j,i)
depth3d, lon3d, lat3d = xr.broadcast(depth, lon, lat)
drF3d, _, _ = xr.broadcast(drF, lon, lat)

# =====================
# xgcm grid for gradients
# =====================
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


# =====================
# Helper functions
# =====================
def compute_Hml(rho_profile, depth_profile, threshold=0.03):
    """Compute mixed layer depth from density profile"""
    rho_10m = rho_profile[6]  # ~10m depth
    mask = rho_profile > rho_10m + threshold
    if not np.any(mask):
        return 0.0
    return float(depth_profile[mask].max())

def ml_integrated_b(b_profile, Hml_value, depth_profile, drF_profile):
    """
    Compute depth-integrated buoyancy over the mixed layer
    for non-uniform vertical grid.
    """
    mask = depth_profile <= Hml_value
    if np.any(mask):
        return float((b_profile[mask] * drF_profile[mask]).sum() / drF_profile[mask].sum())
    else:
        return np.nan

def grad_center(var):
    dx = grid.derivative(var, axis="X")
    dy = grid.derivative(var, axis="Y")
    dx = grid.interp(dx, axis="X", to="center")
    dy = grid.interp(dy, axis="Y", to="center")
    return dx, dy

# =====================
# Compute potential density
# =====================
SA = xr.apply_ufunc(
    gsw.SA_from_SP, salt, depth3d, lon3d, lat3d,
    input_core_dims=[["k","j","i"], ["k","j","i"], ["k","j","i"], ["k","j","i"]],
    output_core_dims=[["k","j","i"]],
    vectorize=True, dask="parallelized", output_dtypes=[float]
)

CT = xr.apply_ufunc(
    gsw.CT_from_pt, SA, theta,
    input_core_dims=[["k","j","i"], ["k","j","i"]],
    output_core_dims=[["k","j","i"]],
    vectorize=True, dask="parallelized", output_dtypes=[float]
)

p_ref = 0
rho = xr.apply_ufunc(
    gsw.rho, SA, CT, p_ref,
    input_core_dims=[["k","j","i"], ["k","j","i"], []],
    output_core_dims=[["k","j","i"]],
    vectorize=True, dask="parallelized", output_dtypes=[float]
)

# =====================
# Compute Mixed Layer Depth (MLD)
# =====================
Hml = xr.apply_ufunc(
    compute_Hml, rho, depth3d,
    input_core_dims=[["k"], ["k"]],
    output_core_dims=[[]],
    vectorize=True, dask="parallelized", output_dtypes=[float]
)

# =====================
# Mixed-layer buoyancy
# =====================
rho_mean_ij = rho.mean(dim=("i","j"))
b = -9.81 * (rho - rho_mean_ij) / 1027.5

B_ml = xr.apply_ufunc(
    ml_integrated_b, b, Hml, depth3d, drF3d,
    input_core_dims=[["k"], [], ["k"], ["k"]],
    output_core_dims=[[]],
    vectorize=True, dask="parallelized", output_dtypes=[float]
)

# =====================
# Compute gradients
# =====================
bx, by = grad_center(B_ml)

# =====================
# Save results
# =====================
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/ml_buoyancy_gradients"
os.makedirs(out_dir, exist_ok=True)

ds_out = xr.Dataset(
    {"Hml": Hml, "B_ml": B_ml, "dbdx": bx, "dbdy": by},
    coords={"lon": lon, "lat": lat, "time": rho.time}
)
ds_out.to_netcdf(f"{out_dir}/ML_buoyancy_gradients_hourly.nc")

print("✓ Saved hourly MLD and mixed-layer buoyancy gradients.")

import os
import numpy as np
import xarray as xr
import gsw
from glob import glob
from xgcm import Grid
from dask.distributed import Client, LocalCluster

# ============================================================
# Setup Dask cluster
# ============================================================
cluster = LocalCluster(
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB",
    processes=True
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ============================================================
# Domain settings
# ============================================================
# from set_constant import domain_name, face, i, j
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ============================================================
# Load LLC4320 base dataset (for grid + coordinates)
# ============================================================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

lon = ds1.XC.isel(face=face, i=i, j=j)
lat = ds1.YC.isel(face=face, i=i, j=j)
depth = ds1.Z
drF = ds1.drF

# Broadcast depth and grid spacing
depth3d, lon3d, lat3d = xr.broadcast(depth, lon, lat)
drF3d, _, _ = xr.broadcast(drF, lon, lat)

# ============================================================
# Setup xgcm grid
# ============================================================
ds_grid_face = ds1.isel(face=face,i=i,j=j,i_g=i, j_g=j,k=0, k_p1=0, k_u=0)

# Drop time if present
if "time" in ds_grid_face.dims:
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

# ============================================================
# Helper functions
# ============================================================
def ml_integrated_b(b_profile, Hml_value, depth_profile, drF_profile):
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

# ============================================================
# Load rho / Hml timeseries as a whole dataset
# ============================================================
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
input_files = sorted(glob(os.path.join(input_dir, "rho_Hml_TS_daily_*.nc")))

print(f"Found {len(input_files)} daily rho/Hml files.")

ds_rho = xr.open_mfdataset(
    input_files,
    parallel=True,
    combine="nested",
    concat_dim="time",
    chunks={"time": 1}
)

rho = ds_rho["rho_daily"]
Hml = ds_rho["Hml_daily"]

# ============================================================
# Compute buoyancy anomaly
# ============================================================
rho_mean_ij = rho.mean(dim=("i", "j"))
b = -9.81 * (rho - rho_mean_ij) / 1027.5  # buoyancy anomaly

# ============================================================
# Compute mixed layer–averaged buoyancy B_ml
# ============================================================
B_ml = xr.apply_ufunc(
    ml_integrated_b,
    b,
    Hml,
    depth3d,
    drF3d,
    input_core_dims=[["k"], [], ["k"], ["k"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
)

# ============================================================
# Compute horizontal gradients of B_ml
# ============================================================
dbdx, dbdy = grad_center(B_ml)

# ============================================================
# Save output to a single file
# ============================================================
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(out_dir, exist_ok=True)

ds_out = xr.Dataset(
    {
        "Hml": Hml,
        "B_ml": B_ml,
        "dbdx": dbdx,
        "dbdy": dbdy
    },
    coords={
        "lon": lon,
        "lat": lat,
        "time": rho.time
    }
)

output_file = f"{out_dir}/ML_buoyancy_gradients_daily.nc"

ds_out = ds_out.compute()
ds_out.to_netcdf(output_file)

print(f"\n✓ Saved mixed-layer buoyancy + gradients to:")
print(output_file)

client.close()
cluster.close()

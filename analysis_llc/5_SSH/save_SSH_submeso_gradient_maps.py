import os
import numpy as np
import xarray as xr
from glob import glob
from xgcm import Grid
from tqdm import tqdm

# ==============================================================
# ========== Domain ==========
# ==============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ==============================================================
# Paths
# ==============================================================
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
eta_dir = os.path.join(base_dir, "surface_24h_avg")

outdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory"
os.makedirs(outdir, exist_ok=True)
out_file = os.path.join(outdir, "eta_submeso_grad_mag_daily.nc")

# ==============================================================
# Load grid info
# ==============================================================
ds1 = xr.open_zarr(
    "/orcd/data/abodner/003/LLC4320/LLC4320",
    consolidated=False
)

lon = ds1["XC"].isel(face=face, i=i, j=j)
lat = ds1["YC"].isel(face=face, i=i, j=j)

ds_grid_face = ds1.isel(
    face=face,
    i=i,
    j=j,
    i_g=i,
    j_g=j,
    k=0,
    k_p1=0,
    k_u=0,
)

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


# ==============================================================
# Load submesoscale SSH
# ==============================================================
shortname = "SSH_Gaussian_submeso_LambdaMLI"
fname = os.path.join(
    base_dir,
    "SSH_submesoscale",
    f"{shortname}.nc",
)

eta_submeso = xr.open_dataset(fname)["SSH_submesoscale"]
eta_submeso = eta_submeso.assign_coords(
    time=eta_submeso.time.dt.floor("D")
)

# ==============================================================
# Helper: gradient magnitude
# ==============================================================
def compute_grad_mag(var, grid):
    var_x = grid.derivative(var, axis="X")
    var_y = grid.derivative(var, axis="Y")

    var_x_c = grid.interp(var_x, axis="X", to="center")
    var_y_c = grid.interp(var_y, axis="Y", to="center")

    grad_mag = np.sqrt(var_x_c**2 + var_y_c**2)
    return grad_mag

# ==============================================================
# Main loop: compute daily maps
# ==============================================================
grad_mag_list = []
times = []

for t in tqdm(range(len(eta_submeso.time)), desc="Computing daily |∇η_submeso|"):
    eta_t = eta_submeso.isel(time=t)

    grad_mag_t = compute_grad_mag(eta_t, grid)

    grad_mag_list.append(grad_mag_t)
    times.append(eta_submeso.time.isel(time=t).values)

# ==============================================================
# Save to NetCDF
# ==============================================================
eta_submeso_grad_mag = xr.concat(grad_mag_list, dim="time")
eta_submeso_grad_mag = eta_submeso_grad_mag.assign_coords(
    time=("time", times),
    lon=lon,
    lat=lat,
)

ds_out = xr.Dataset(
    {
        "eta_submeso_grad_mag": eta_submeso_grad_mag
    }
)

ds_out.to_netcdf(out_file)

print(f"✅ Saved daily eta_submeso_grad_mag maps to:")
print(out_file)

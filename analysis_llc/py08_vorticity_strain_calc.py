##############
############## Compute surface strain rate, surface vorticity, and surface divergence
##############

import xarray as xr
import numpy as np
import os
from xgcm import Grid

from set_constant import domain_name, face, i, j

# ========= Paths =========
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
uv_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
output_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/strain_vorticity"
os.makedirs(output_path, exist_ok=True)

# ========= Load grid data =========
print("Loading grid...")
ds1 = xr.open_zarr(grid_path, consolidated=False)
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

# ========= Load daily averaged U and V =========
print("Loading daily averaged U and V...")
u_path = os.path.join(uv_dir, "uu_s_24h_*.nc")
v_path = os.path.join(uv_dir, "vv_s_24h_*.nc")

dsu = xr.open_mfdataset(u_path, combine='by_coords')
dsv = xr.open_mfdataset(v_path, combine='by_coords')

# Align datasets and select face/i/j region
U = dsu["U"]
V = dsv["V"]

# Align time
U, V = xr.align(U, V, join="inner")

# ========= Compute derivatives =========
print("Computing derivatives...")

# ∂u/∂x at center
u_x = grid.derivative(U, axis="X")  # center

# ∂v/∂y at center
v_y = grid.derivative(V, axis="Y")  # center

# ∂v/∂x and ∂u/∂y for vorticity and shear strain
v_x = grid.derivative(V, axis="X")
u_y = grid.derivative(U, axis="Y")

# ========= Derived fields =========
print("Computing strain, divergence, and vorticity...")

sigma_n = u_x - v_y
sigma_s = u_y + v_x

# strain magnitude at center
sigma_s_center = grid.interp(sigma_s, axis="X", to="center")
sigma_s_center = grid.interp(sigma_s_center, axis="Y", to="center")
strain_mag = np.sqrt(sigma_n**2 + sigma_s_center**2)

divergence = u_x + v_y
vorticity = v_x - u_y

sigma_n = sigma_n.assign_coords(time=U.time)
sigma_s = sigma_s.assign_coords(time=U.time)
strain_mag = strain_mag.assign_coords(time=U.time)
divergence = divergence.assign_coords(time=U.time)
vorticity = vorticity.assign_coords(time=U.time)

# ========= Save results =========
print("Saving results...")

ds_out = xr.Dataset({
    "sigma_n": sigma_n,
    "sigma_s": sigma_s,
    "strain_mag": strain_mag,
    "divergence": divergence,
    "vorticity": vorticity
})

ds_out.to_netcdf(os.path.join(output_path, "strain_vorticity_daily.nc"))

print("Done: strain and vorticity fields saved.")



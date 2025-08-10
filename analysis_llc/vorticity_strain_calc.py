##############
############## Compute surface strain rate, surface vorticity, and surface divergence
##############

import xarray as xr
import numpy as np
import os
from xgcm import Grid

# ========= Paths =========
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
uv_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/surfaceUV_24h_avg"
output_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/strain_vort"
os.makedirs(output_path, exist_ok=True)

# ========= Domain =========
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ========= Load grid data =========
print("Loading grid...")
ds_grid = xr.open_zarr(grid_path, consolidated=False)
ds_grid_face = ds_grid.sel(face=face).isel(i=i, j=j)

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

# ========= Load daily U and V =========
print("Loading daily U and V...")
u_path = os.path.join(uv_dir, "uu_s_24h_*.nc")
v_path = os.path.join(uv_dir, "vv_s_24h_*.nc")

dsu = xr.open_mfdataset(u_path, combine='by_coords')
dsv = xr.open_mfdataset(v_path, combine='by_coords')

# Align datasets and select face/i/j region
U = dsu["U"].sel(i_g=i, j=j)
V = dsv["V"].sel(i=i, j_g=j)

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
strain_mag = np.sqrt(sigma_n**2 + sigma_s**2)
divergence = u_x + v_y
vorticity = v_x - u_y

# ========= Save results =========
print("Saving results...")

ds_out = xr.Dataset({
    "sigma_n": sigma_n,
    "sigma_s": sigma_s,
    "strain_magnitude": strain_mag,
    "divergence": divergence,
    "vorticity": vorticity
})

ds_out.to_netcdf(os.path.join(output_path, "strain_vorticity_daily.nc"))

print("Done: strain and vorticity fields saved.")



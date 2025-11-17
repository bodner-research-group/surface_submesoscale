# #!/usr/bin/env python3
# # ==============================================================
# # Compute and save gradient magnitude and Laplacian timeseries
# # for SSH_Gaussian_meso_30kmCutoff_1_12deg.nc
# # Following methodology in Wang et al. (2025)
# # ==============================================================

# import os
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# from xgcm import Grid
# from tqdm import tqdm

# # ==============================================================
# # Domain constants (from set_constant.py)
# # ==============================================================
# domain_name = "icelandic_basin"
# face = 2
# i = slice(527, 1007)   # icelandic_basin -- larger domain
# j = slice(2960, 3441)  # icelandic_basin -- larger domain

# # ==============================================================
# # Paths
# # ==============================================================
# base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
# ssh_file = os.path.join(base_dir, "SSH_submesoscale", "SSH_Gaussian_meso_30kmCutoff_1_12deg.nc")

# out_dir = os.path.join(base_dir, "grad_laplace_timeseries_meso_30kmCutoff_1_12deg")
# os.makedirs(out_dir, exist_ok=True)

# figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_meso_grad/"
# os.makedirs(figdir, exist_ok=True)

# # ==============================================================
# # Load grid information from LLC4320
# # ==============================================================
# ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
# lon = ds1["XC"].isel(face=face, i=i, j=j)
# lat = ds1["YC"].isel(face=face, i=i, j=j)

# # Subsample grid to match coarse 1/12° resolution
# coarse_factor = 4
# lon_coarse = lon.coarsen(i=coarse_factor, j=coarse_factor, boundary="trim").mean()
# lat_coarse = lat.coarsen(i=coarse_factor, j=coarse_factor, boundary="trim").mean()

# ds_grid_face = ds1.isel(face=face, i=i, j=j, i_g=i, j_g=j, k=0, k_p1=0, k_u=0)
# if "time" in ds_grid_face.dims:
#     ds_grid_face = ds_grid_face.isel(time=0, drop=True)

# coords = {"X": {"center": "i", "left": "i_g"}, "Y": {"center": "j", "left": "j_g"}}
# metrics = {("X",): ["dxC", "dxG"], ("Y",): ["dyC", "dyG"]}
# grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)

# # ==============================================================
# # Load mesoscale coarse SSH
# # ==============================================================
# print("🔹 Loading mesoscale coarse SSH...")
# ds_ssh = xr.open_dataset(ssh_file)
# SSH_meso = ds_ssh["SSH_mesoscale_coarse"]
# SSH_meso = SSH_meso.assign_coords(time=SSH_meso.time.dt.floor("D"))
# print("✅ SSH loaded with shape:", SSH_meso.shape)

# # ==============================================================
# # Helper function
# # ==============================================================
# def compute_grad_laplace(var, grid):
#     """Compute gradient magnitude and Laplacian using xgcm grid."""
#     var_x = grid.derivative(var, axis="X")
#     var_y = grid.derivative(var, axis="Y")

#     var_x_c = grid.interp(var_x, axis="X", to="center")
#     var_y_c = grid.interp(var_y, axis="Y", to="center")
#     grad_mag = np.sqrt(var_x_c**2 + var_y_c**2)

#     var_xx = grid.derivative(var_x_c, axis="X")
#     var_yy = grid.derivative(var_y_c, axis="Y")

#     var_xx_c = grid.interp(var_xx, axis="X", to="center")
#     var_yy_c = grid.interp(var_yy, axis="Y", to="center")
#     laplace = var_xx_c + var_yy_c

#     return grad_mag, laplace

# # ==============================================================
# # Loop over time steps
# # ==============================================================
# times = []
# grad2_mean_list = []

# for t in tqdm(range(len(SSH_meso.time)), desc="Processing mesoscale SSH"):
#     time_val = SSH_meso.time.isel(time=t).values
#     date_tag = np.datetime_as_string(time_val, unit="D").replace("-", "")

#     ssh_t = SSH_meso.isel(time=t)
#     ssh_t = ssh_t - ssh_t.mean(dim=["i", "j"])  # remove domain mean

#     grad_mag, laplace = compute_grad_laplace(ssh_t, grid)

#     grad2_mean = (grad_mag.isel(i=slice(2, -2), j=slice(2, -2))**2).mean(dim=["i", "j"])
#     grad2_mean_list.append(grad2_mean)
#     times.append(time_val)

# # ==============================================================
# # Save domain-mean |∇η|² timeseries
# # ==============================================================
# ts_ds = xr.Dataset(
#     {
#         "SSH_meso_grad2_mean": xr.concat(grad2_mean_list, dim="time"),
#     },
#     coords={"time": ("time", times)},
# )

# out_path = os.path.join(out_dir, "grad2_meso_30kmCutoff_1_12deg_timeseries.nc")
# ts_ds.to_netcdf(out_path)
# print(f"✅ Saved domain-mean |∇η|² timeseries: {out_path}")

# # ==============================================================
# # Plot timeseries
# # ==============================================================
# plt.figure(figsize=(8, 4))
# ts_ds["SSH_meso_grad2_mean"].plot(label="⟨mesoscale |∇η|²⟩", color="tab:blue")

# # 7-day rolling mean
# ts_ds["SSH_meso_grad2_mean"].compute().rolling(time=7, center=True).mean().plot(
#     label="⟨mesoscale |∇η|²⟩ (7d mean)", color="tab:blue", linestyle="--"
# )

# plt.title("Domain-mean mesoscale |∇η|² Timeseries (30 km cutoff, 1/12° grid)")
# plt.ylabel("Mean(|∇η|²) [m²/m²]")
# plt.xlabel("Time")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()

# fig_path = os.path.join(figdir, "grad2_meso_30kmCutoff_1_12deg_timeseries.png")
# plt.savefig(fig_path, dpi=150)
# plt.close()
# print(f"✅ Saved figure: {fig_path}")

# print("🏁 Done.")





#!/usr/bin/env python3
# ==============================================================
# Compute and save gradient magnitude and Laplacian timeseries
# for SSH_Gaussian_meso_30kmCutoff_1_12deg.nc
# Following Wang et al. (2025) methodology
# ==============================================================

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================
# Domain constants
# ==============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ==============================================================
# Paths
# ==============================================================
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
ssh_file = os.path.join(base_dir, "SSH_submesoscale", "SSH_Gaussian_meso_30kmCutoff_1_12deg.nc")

out_dir = os.path.join(base_dir, "grad_laplace_timeseries_meso_30kmCutoff_1_12deg")
os.makedirs(out_dir, exist_ok=True)

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_meso_grad/"
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Load mesoscale coarse SSH
# ==============================================================
print("🔹 Loading mesoscale coarse SSH...")
ds_ssh = xr.open_dataset(ssh_file)
SSH_meso = ds_ssh["SSH_mesoscale_coarse"]
SSH_meso = SSH_meso.assign_coords(time=SSH_meso.time.dt.floor("D"))
print("✅ SSH loaded with shape:", SSH_meso.shape)

# ==============================================================
# Load grid spacing info from LLC4320
# ==============================================================
print("🔹 Loading grid info from LLC4320 to compute dx, dy ...")
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

dxC_mean = ds1.dxC.isel(face=face, i_g=i, j=j).values.mean() / 1000.0  # km
dyC_mean = ds1.dyC.isel(face=face, i=i, j_g=j).values.mean() / 1000.0  # km
dx_km = np.sqrt(0.5 * (dxC_mean**2 + dyC_mean**2))
nyquist_wavelength = 2 * dx_km
print(f"Grid spacing (native LLC4320): {dx_km:.2f} km, Nyquist λ = {nyquist_wavelength:.2f} km")

# coarse factor (1/48° → 1/12°)
coarse_factor = 4
dx_km_coarse = dx_km * coarse_factor
dy_km_coarse = dx_km_coarse
dx = dx_km_coarse * 1000  # m
dy = dy_km_coarse * 1000  # m
print(f"Effective coarse grid spacing: {dx_km_coarse:.2f} km")

# ==============================================================
# Helper function
# ==============================================================
def compute_gradients(var, dx, dy):
    """Compute gradient magnitude and Laplacian using finite differences."""
    dvar_dy, dvar_dx = np.gradient(var, dy, dx)
    grad_mag = np.sqrt(dvar_dx**2 + dvar_dy**2)
    laplace = (
        np.gradient(dvar_dx, dx, axis=-1)
        + np.gradient(dvar_dy, dy, axis=-2)
    )
    return grad_mag, laplace

# ==============================================================
# Main loop over time
# ==============================================================
times = []
grad2_mean_list = []

for t in tqdm(range(len(SSH_meso.time)), desc="Processing mesoscale SSH"):
    ssh_t = SSH_meso.isel(time=t)
    ssh_t = ssh_t - ssh_t.mean(dim=["i", "j"])  # remove domain mean

    grad_mag, laplace = compute_gradients(ssh_t.values, dx, dy)
    grad2_mean = np.nanmean(grad_mag[2:-2, 2:-2] ** 2)

    grad2_mean_list.append(grad2_mean)
    times.append(SSH_meso.time.isel(time=t).values)

# ==============================================================
# Save domain-mean |∇η|² timeseries
# ==============================================================
ts_ds = xr.Dataset(
    {"SSH_meso_grad2_mean": (("time",), grad2_mean_list)},
    coords={"time": ("time", times)},
)
out_path = os.path.join(out_dir, "grad2_meso_30kmCutoff_1_12deg_timeseries.nc")
ts_ds.to_netcdf(out_path)
print(f"✅ Saved domain-mean |∇η|² timeseries: {out_path}")

# ==============================================================
# Plot timeseries
# ==============================================================
plt.figure(figsize=(8, 4))
ts_ds["SSH_meso_grad2_mean"].plot(label="⟨mesoscale |∇η|²⟩", color="tab:blue")

# 7-day rolling mean
ts_ds["SSH_meso_grad2_mean"].rolling(time=7, center=True).mean().plot(
    label="⟨mesoscale |∇η|²⟩ (7d mean)", color="tab:blue", linestyle="--"
)

plt.title("Domain-mean mesoscale |∇η|² Timeseries (30 km cutoff, 1/12° grid)")
plt.ylabel("Mean(|∇η|²) [m²/m²]")
plt.xlabel("Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

fig_path = os.path.join(figdir, "grad2_meso_30kmCutoff_1_12deg_timeseries.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"✅ Saved figure: {fig_path}")

print("🏁 Done.")

import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
from xgcm import Grid
from dask.diagnostics import ProgressBar
from scipy.io import savemat

from set_constant import domain_name, face, i, j
from set_colormaps import WhiteBlueGreenYellowRed
cmap = WhiteBlueGreenYellowRed()

# from dask.distributed import Client, LocalCluster

# # Dask Cluster Setup
# cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
# client = Client(cluster)
# print("✅ Dask cluster started")
# print("Dask dashboard:", client.dashboard_link)


# --- Physical constants ---
g = 9.81        # gravity (m/s²)
rho0 = 1025     # reference density (kg/m³)
omega = 7.2921e-5  # Earth rotation rate

# --- Paths ---
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
hml_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_weekly"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Lambda_MLI"
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI"
hml_files = sorted(glob(os.path.join(hml_dir, "rho_Hml_TS_7d_*.nc")))

os.makedirs(figdir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# --- Load Grid & Setup xgcm ---
ds1 = xr.open_zarr(grid_path, consolidated=False)

# ds_grid_face = ds1.isel(face=face,i=i, j=j,i_g=i, j_g=j,k=0,k_p1=0,k_u=0)
ds_grid_face = ds1.isel(face=face,i=i, j=j,i_g=i, j_g=j,k_p1=0,k_u=0)

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


lat2d = ds1.YC.isel(face=face, i=i, j=j)
lon2d = ds1.XC.isel(face=face, i=i, j=j)
depth = ds1.Z.values

# lon_plot = lon2d.transpose("j", "i").values[:-1, :-1]
# lat_plot = lat2d.transpose("j", "i").values[:-1, :-1]

lon_plot = lon2d.transpose("j", "i")
lat_plot = lat2d.transpose("j", "i")

# --- Coriolis parameter ---
f_cor = 2 * omega * np.sin(np.deg2rad(lat2d.values))
f_cor_squared = f_cor**2





# --- Helper function to compute N² ---
def compute_N2_xr(rho, depth):
    drho = - rho.differentiate("k")
    dz = - depth.differentiate("k")
    N2 = - (g / rho0) * (drho / dz)
    return N2

# --- Loop through files ---
for fpath in hml_files:
# fpath = hml_files[9]

    print(f"Processing {os.path.basename(fpath)}")
    ds = xr.open_dataset(fpath)
    date_tag = os.path.basename(fpath).split("_")[-1].replace(".nc", "")

    Hml = ds["Hml_7d"].load()
    rho = ds["rho_7d"].load()  # (k, j, i)


    # Depth broadcast
    depth_broadcasted = xr.DataArray(
        np.broadcast_to(depth[:, None, None], rho.shape),
        dims=rho.dims,
        coords=rho.coords
    )

    # Compute N²
    N2 = compute_N2_xr(rho, depth_broadcasted)

    # Find index of Hml base
    k_hml_base = np.abs(depth[:, None, None] - Hml.values[None, :, :]).argmin(axis=0)
    k_hml_base_da = xr.DataArray(k_hml_base, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})

    # Create mask
    # k_indices, _, _ = xr.broadcast(N2["k"], N2["j"], N2["i"])
    # k_indices = k_indices.astype(int)
    # in_range_mask = k_indices <= k_hml_base_da
    # N2_masked = N2.where(in_range_mask)
    # N2ml_mean = N2_masked.mean(dim="k", skipna=True)

    # Compute depth bounds (30%-90%) of mixed layer
    Hml_30 = Hml * 0.3
    Hml_90 = Hml * 0.9
    # Interpolate to nearest depth levels
    k_30 = np.abs(depth[:, None, None] - Hml_30.values[None, :, :]).argmin(axis=0)
    k_90 = np.abs(depth[:, None, None] - Hml_90.values[None, :, :]).argmin(axis=0)
    # Convert to xarray DataArrays
    k_30_da = xr.DataArray(k_30, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})
    k_90_da = xr.DataArray(k_90, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})
    # Create k indices aligned with N2
    k_indices, _, _ = xr.broadcast(N2["k"], N2["j"], N2["i"])
    k_indices = k_indices.astype(int)
    # Mask for k within 30%-90% of Hml
    in_range_mask = (k_indices >= k_30_da) & (k_indices <= k_90_da)
    # Apply mask
    N2_masked = N2.where(in_range_mask)
    # Mean over vertical dimension (k)
    N2ml_mean = N2_masked.mean(dim="k", skipna=True)

    # --- Compute horizontal buoyancy gradient squared Mml² using xgcm ---
    rho_x = grid.derivative(rho, axis="X")
    rho_y = grid.derivative(rho, axis="Y")
    rho_x_center = grid.interp(rho_x, axis="X", to="center")
    rho_y_center = grid.interp(rho_y, axis="Y", to="center")
    M4_full = ((g / rho0) ** 2) * (rho_x_center**2 + rho_y_center**2)
    M4_masked = M4_full.where(in_range_mask)
    Mml4_mean = M4_masked.mean(dim="k", skipna=True)

    # --- Compute Rib and Lambda_MLI ---
    f_cor_3D_squared = xr.DataArray(
        np.broadcast_to(f_cor_squared[None, :, :], rho.shape),
        dims=rho.dims,
        coords=rho.coords
    )

    Rib_local = (N2 * f_cor_3D_squared) / M4_full
    Rib_local = Rib_local.where(Rib_local > 0)
    Rib_masked = Rib_local.where(in_range_mask)
    Rib = Rib_masked.mean(dim="k", skipna=True)


    Lambda_MLI = (2 * np.pi / np.sqrt(5 / 2)) * np.sqrt(1 + 1 / Rib) * np.sqrt(N2ml_mean) * np.abs(Hml) / f_cor

    # --- Plot ---
    # plt.figure(figsize=(8, 6))
    # plt.pcolormesh(lon_plot, lat_plot, Lambda_MLI/1000, cmap=cmap, shading="auto",vmin = 0, vmax = 20)
    # plt.colorbar(label="Lambda_MLI (km)")
    # plt.title(f"MLI Wavelength - {date_tag}")
    # plt.tight_layout()
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.savefig(os.path.join(figdir, f"Lambda_MLI_{date_tag}.png"), dpi=150)
    # plt.close()

    # --- 2x2 Plot of Lambda_MLI, N2ml_mean, Hml, Rib ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plots = [
        (Lambda_MLI / 1000, "Lambda_MLI (km)", (0, 40), cmap),
        (N2ml_mean, "N² (s⁻²)", (0, 2e-6), "viridis"),
        # (Hml, "Hml (m)", None, "plasma"),
        (np.sqrt(Mml4_mean), "M2", (0, 5e-8), "plasma"),
        (Rib, "Ri_b", (0, 10), "magma"),
    ]

    for ax, (data, label, clim, cm) in zip(axs.flat, plots):
        p = ax.pcolormesh(lon_plot, lat_plot, data, shading="auto", cmap=cm)
        if clim:
            p.set_clim(*clim)
        cbar = plt.colorbar(p, ax=ax, orientation="vertical")
        cbar.set_label(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(label)

    plt.suptitle(f"MLI Diagnostics - {date_tag}", fontsize=16)
    plot_path = os.path.join(figdir, f"MLI_summary_{date_tag}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()


    # --- Save to NetCDF ---
    ds_out = xr.Dataset(
        {
            "Lambda_MLI": Lambda_MLI,
            "Rib": Rib,
            "N2ml_mean": N2ml_mean,
            "Mml4_mean": Mml4_mean,
            "Hml": Hml,
            "f_cor": (("j", "i"), f_cor),  # make it compatible with j/i dims
        }
    )

    out_nc = os.path.join(output_dir, f"Lambda_MLI_{date_tag}.nc")

    # Optional: Compression
    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds_out.data_vars
    }

    ds_out.to_netcdf(out_nc, encoding=encoding)

    print(f"✅ Saved NetCDF to {out_nc}")

print("✅ All done.")

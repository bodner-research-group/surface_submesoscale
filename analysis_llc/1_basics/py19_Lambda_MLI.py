import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
from xgcm import Grid
from dask.diagnostics import ProgressBar
from scipy.io import savemat
import gsw

from set_colormaps import WhiteBlueGreenYellowRed
cmap = WhiteBlueGreenYellowRed()

# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

from dask.distributed import Client, LocalCluster

# Dask Cluster Setup
cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit="11GB")
# cluster = LocalCluster(n_workers=20, threads_per_worker=1, memory_limit="17.6GB")
client = Client(cluster)
print("✅ Dask cluster started")
print("Dask dashboard:", client.dashboard_link)


# --- Physical constants ---
g = 9.81        # gravity (m/s²)
rho0 = 1027.5     # reference density (kg/m³)
omega = 7.2921e-5  # Earth rotation rate

# --- Paths ---
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
# hml_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_7d_rolling_mean"
# figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Lambda_MLI"
# output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI"
# hml_files = sorted(glob(os.path.join(hml_dir, "rho_Hml_TS_7d_*.nc")))
hml_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Lambda_MLI_daily_surface_reference_GSW"
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_daily_surface_reference_GSW"
# hml_files = sorted(glob(os.path.join(hml_dir, "rho_Hml_TS_daily_*.nc")))
hml_files = sorted(glob(os.path.join(hml_dir, "Hml_daily_surface_reference_*.nc")))

os.makedirs(figdir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# --- Load Grid & Setup xgcm ---
ds1 = xr.open_zarr(grid_path, consolidated=False)
drF = ds1.drF  # vertical grid spacing, 1D
lon = ds1['XC'].isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
lat = ds1['YC'].isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
drF3d, _, _ = xr.broadcast(drF, lon, lat)

# ds_grid_face = ds1.isel(face=face,i=i, j=j,i_g=i, j_g=j,k=0,k_p1=0,k_u=0)
ds_grid_face = ds1.isel(face=face,i=i, j=j,i_g=i, j_g=j,k_u=0)

# Drop time dimension if exists
if 'time' in ds_grid_face.dims:
    ds_grid_face = ds_grid_face.isel(time=0, drop=True)  # or .squeeze('time')

# ========= Setup xgcm grid =========
# coords = {
#     "X": {"center": "i", "left": "i_g"},
#     "Y": {"center": "j", "left": "j_g"},
# }
# metrics = {
#     ("X",): ["dxC", "dxG"],
#     ("Y",): ["dyC", "dyG"],
# }
# grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)


coords = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
    "Z": {"center": "k", "outer": "k_p1"}, 
}
metrics = {
    ("X",): ["dxC", "dxG"],
    ("Y",): ["dyC", "dyG"],
    ("Z",): ["drF"],  
}
grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)


lat = ds1.YC.isel(face=face, i=i, j=j)
lon = ds1.XC.isel(face=face, i=i, j=j)
depth = ds1.Z.values
depth_full = ds1.Z

# ========== Broadcast lon, lat, depth ==========
depth_for_broadcast = ds1.Z
depth3d, _, _ = xr.broadcast(np.abs(depth_for_broadcast), lon, lat)
depth3d = depth3d.chunk({"k": -1, "j": -1, "i": -1})

# lon_plot = lon.transpose("j", "i").values[:-1, :-1]
# lat_plot = lat.transpose("j", "i").values[:-1, :-1]

lon_plot = lon.transpose("j", "i")
lat_plot = lat.transpose("j", "i")

# --- Coriolis parameter ---
f_cor = 2 * omega * np.sin(np.deg2rad(lat.values))
f_cor_squared = f_cor**2

# --- Helper function to compute N² ---
# def compute_N2_xr(rho, depth):
#     drho = - rho.differentiate("k")
#     dz = - depth.differentiate("k")
#     N2 = - (g / rho0) * (drho / dz)
#     return N2

#### Incorrect: use GSW toolbox to compute N^2
# def compute_N2_xr(rho, depth, grid):
#     drho_dz = grid.derivative(rho, axis="Z") / grid.derivative(depth, axis="Z")
#     N2 = - (g / rho0) * drho_dz
#     # Optional: interp to center
#     N2_centered = grid.interp(N2, axis="Z", to="center")
#     return N2_centered


# --- Loop through files ---
for fpath in hml_files:
# for fpath in hml_files[::-1]:
# fpath = hml_files[0]

    print(f"Processing {os.path.basename(fpath)}")
    ds = xr.open_dataset(fpath)
    date_tag = os.path.basename(fpath).split("_")[-1].replace(".nc", "")
    out_nc = os.path.join(output_dir, f"Lambda_MLI_{date_tag}.nc")

    if os.path.exists(out_nc):
        print(f"⏭️  Skipping {date_tag}, output already exists.")
        continue

    Hml = ds["Hml_daily"].load()

    raw_file = f"{hml_dir}/rho_Hml_TS_daily_{date_tag}.nc"
    ds_rho = xr.open_dataset(raw_file)
    rho = ds_rho["rho_daily"].load()  # (k, j, i)

    T_daily = ds_rho["T_daily"].load()
    S_daily = ds_rho["S_daily"].load()

    # Ensure vertical is single chunk and load into memory
    S_daily = S_daily.chunk({"k": -1})
    T_daily = T_daily.chunk({"k": -1})

    # --- Compute pressure ---
    pressure = gsw.p_from_z(-depth3d, lat.values)

    # --- Compute Absolute Salinity ---
    SA = gsw.SA_from_SP(S_daily.values, pressure, lon.values, lat.values)

    # --- Compute Conservative Temperature ---
    CT = gsw.CT_from_pt(SA, T_daily.values)

    # --- Compute N² along vertical ---
    N2_array, p_mid = gsw.Nsquared(SA, CT, pressure, lat=lat.values, axis=0)

    # --- Wrap as xarray DataArray using k index ---
    N2 = xr.DataArray(
        N2_array,
        dims=("k", "j", "i"),
        coords={
            "k": np.arange(N2_array.shape[0]),  # keep simple 0..k-1 index
            "j": lat.j,
            "i": lat.i
        },
        name="N2",
        attrs={
            "long_name": "Buoyancy frequency squared",
            "units": "s^-2"
        }
    )

    # Find index of Hml base
    k_hml_base = np.abs(depth[:, None, None] - Hml.values[None, :, :]).argmin(axis=0)
    k_hml_base_da = xr.DataArray(k_hml_base, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})

    ### Compute depth bounds (0%-100%) of mixed layer
    ### Create mask
    k_indices, _, _ = xr.broadcast(N2["k"], N2["j"], N2["i"])
    k_indices = k_indices.astype(int)
    in_range_mask = k_indices <= k_hml_base_da

    ### Compute depth bounds (50%-90%) of mixed layer
    # Hml_upper = Hml * 0.5
    # Hml_lower = Hml * 0.9
    # # Interpolate to nearest depth levels
    # k_upper = np.abs(depth[:, None, None] - Hml_upper.values[None, :, :]).argmin(axis=0)
    # k_lower = np.abs(depth[:, None, None] - Hml_lower.values[None, :, :]).argmin(axis=0)
    # # Convert to xarray DataArrays
    # k_upper_da = xr.DataArray(k_upper, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})
    # k_lower_da = xr.DataArray(k_lower, dims=("j", "i"), coords={"j": rho.j, "i": rho.i})
    # # Create k indices aligned with N2
    # k_indices, _, _ = xr.broadcast(N2["k"], N2["j"], N2["i"])
    # k_indices = k_indices.astype(int)
    # # Mask for k within 50%-90% of Hml
    # in_range_mask = (k_indices >= k_upper_da) & (k_indices <= k_lower_da)

    # Apply mask
    N2_masked = N2.where(in_range_mask)
    # Mean over vertical dimension (k)
    weighted_sum_N2 = (N2_masked * drF3d.where(in_range_mask)).sum(dim="k", skipna=True)
    total_thickness = drF3d.where(in_range_mask).sum(dim="k", skipna=True)
    N2ml_mean = weighted_sum_N2 / total_thickness

    # --- Compute horizontal buoyancy gradient squared Mml² using xgcm ---
    rho_x = grid.derivative(rho, axis="X")
    rho_y = grid.derivative(rho, axis="Y")
    rho_x_center = grid.interp(rho_x, axis="X", to="center")
    rho_y_center = grid.interp(rho_y, axis="Y", to="center")
    M4_full = ((g / rho0) ** 2) * (rho_x_center**2 + rho_y_center**2)
    M4_masked = M4_full.where(in_range_mask)
    Mml4_mean = (M4_masked * drF3d.where(in_range_mask)).sum(dim="k", skipna=True) / total_thickness
    Mml2_mean = (np.sqrt(M4_masked) * drF3d.where(in_range_mask)).sum(dim="k", skipna=True)/total_thickness

    # --- Compute Rib and Lambda_MLI ---
    f_cor_3D_squared = xr.DataArray(
        np.broadcast_to(f_cor_squared[None, :, :], rho.shape),
        dims=rho.dims,
        coords=rho.coords
    )

    ### Compute local balanced Richardson Number
    Rib_local = (N2 * f_cor_3D_squared) / M4_full
    Rib_local = Rib_local.where(Rib_local > 0)
    # Rib_masked = Rib_local.where(in_range_mask)
    # Rib = Rib_masked.mean(dim="k", skipna=True) ### incorrect

    ### Compute the harmonic mean of balanced Richardson Number (to highlight the influence of small local Richardson numbers)
    inverseRib_masked = 1/Rib_local
    inverseRib_masked = inverseRib_masked.where(in_range_mask)
    weighted_sum_inverseRib = (inverseRib_masked * drF3d.where(in_range_mask)).sum(dim="k", skipna=True)
    inverseRib = weighted_sum_inverseRib/total_thickness
    Rib = 1/inverseRib

    Lambda_MLI = (2 * np.pi / np.sqrt(5 / 2)) * np.sqrt(1 + 1 / Rib) * np.sqrt(N2ml_mean) * np.abs(Hml) / np.abs(f_cor)

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

    # # --- 2x2 Plot of Lambda_MLI, N2ml_mean, Hml, Rib ---
    # fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    # plots = [
    #     (Lambda_MLI / 1000, "Lambda_MLI (km)", (0, 40), cmap),
    #     (N2ml_mean, "N² (s⁻²)", (0, 2e-6), "viridis"),
    #     # (Hml, "Hml (m)", None, "plasma"),
    #     (np.sqrt(Mml4_mean), "M2", (0, 5e-8), "plasma"),
    #     (Rib, "Ri_b", (0, 10), "magma"),
    # ]

    # for ax, (data, label, clim, cm) in zip(axs.flat, plots):
    #     p = ax.pcolormesh(lon_plot, lat_plot, data, shading="auto", cmap=cm)
    #     if clim:
    #         p.set_clim(*clim)
    #     cbar = plt.colorbar(p, ax=ax, orientation="vertical")
    #     cbar.set_label(label)
    #     ax.set_xlabel("Longitude")
    #     ax.set_ylabel("Latitude")
    #     ax.set_title(label)

    # plt.suptitle(f"MLI Diagnostics - {date_tag}", fontsize=16)
    # plot_path = os.path.join(figdir, f"MLI_summary_{date_tag}.png")
    # plt.savefig(plot_path, dpi=150)
    # plt.close()


    # --- Save to NetCDF ---
    ds_out = xr.Dataset(
        {
            "Lambda_MLI": Lambda_MLI,
            "Rib": Rib,
            "N2ml_mean": N2ml_mean,
            "Mml4_mean": Mml4_mean,
            "Mml2_mean": Mml2_mean,
            "Hml": Hml,
            "f_cor": (("j", "i"), f_cor),  # make it compatible with j/i dims
        }
    )



    # Optional: Compression
    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds_out.data_vars
    }

    ds_out.to_netcdf(out_nc, encoding=encoding)

    print(f"✅ Saved NetCDF to {out_nc}")



print("✅ All done.")

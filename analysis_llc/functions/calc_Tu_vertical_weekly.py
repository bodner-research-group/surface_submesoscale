# Modified: Compute vertical Turner angle from weekly rho, alpha, beta outputs

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from glob import glob

from set_constant import domain_name, face, i, j

# ========== Paths ==========
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
rho_input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_weekly"
ts_input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle"
os.makedirs(out_dir, exist_ok=True)

# ========== Grid Info ==========
print("Loading grid...")
ds_grid = xr.open_zarr(zarr_path, consolidated=False)
depth = ds_grid.Z
lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)

# ========== Loop over weekly files ==========
rho_files = sorted(glob(os.path.join(rho_input_dir, "rho_Hml_TS_7d_*.nc")))

for rho_file in rho_files:
    date_tag = os.path.basename(rho_file).split("_")[-1].replace(".nc", "")
    print(f"\nProcessing Turner angle for week: {date_tag}")

    # Read rho, alpha, beta
    ds_rho = xr.open_dataset(rho_file)
    rho = ds_rho["rho_7d"]
    alpha = ds_rho["alpha_7d"]
    beta = ds_rho["beta_7d"]
    Hml = ds_rho["Hml_7d"]
    T_7d = ds_rho["T_7d"]
    S_7d = ds_rho["S_7d"]


    # =========================
    # Compute Turner angle
    # =========================

    rho_10m = rho.isel(k=6)
    drho = rho - rho_10m.expand_dims(k=rho.k)

    drho = drho.assign_coords(k=depth)


    Hml_50 = Hml * 0.5
    Hml_90 = Hml * 0.9

    # Compute depth slices
    depth_vals = depth.values
    depth_3d = depth_vals[:, None, None]  # (k, 1, 1)
    Hml_50_3d = Hml_50.values[None, :, :]
    Hml_90_3d = Hml_90.values[None, :, :]

    k_50 = np.abs(depth_3d - Hml_50_3d).argmin(axis=0)
    k_90 = np.abs(depth_3d - Hml_90_3d).argmin(axis=0)

    j_dim, i_dim = k_50.shape
    j_idx, i_idx = np.meshgrid(np.arange(j_dim), np.arange(i_dim), indexing='ij')

    def extract_at_k(arr, k_idx):
        arr_np = arr.values
        return arr_np[k_idx, j_idx, i_idx]

    tt_k50 = extract_at_k(T_7d, k_50)
    tt_k90 = extract_at_k(T_7d, k_90)
    ss_k50 = extract_at_k(S_7d, k_50)
    ss_k90 = extract_at_k(S_7d, k_90)
    alpha_k50 = extract_at_k(alpha, k_50)
    beta_k50 = extract_at_k(beta, k_50)

    dz = depth_vals[k_50] - depth_vals[k_90]
    dz = np.where(dz == 0, np.nan, dz)  # prevent div/0

    dT_dz = (tt_k50 - tt_k90) / dz
    dS_dz = (ss_k50 - ss_k90) / dz

    x = beta_k50 * dS_dz
    y = alpha_k50 * dT_dz

    # Compute Turner angle (in degrees)
    with np.errstate(divide='ignore', invalid='ignore'):
        TuV_rad = np.arctan2((y + x), (y - x))
        TuV_deg = np.degrees(TuV_rad)

    # ========== Kernel Density ==========
    TuV_clean = TuV_deg.values.ravel()[~np.isnan(TuV_deg.values.ravel())]

    if TuV_clean.size < 100:
        print(f"Too few valid Turner angle values for {date_tag}, skipping KDE.")
        continue

    kde = gaussian_kde(TuV_clean, bw_method=0.05)
    x_grid = np.linspace(-180, 180, 1000)
    pdf_values = kde(x_grid)

    # ========== Save ==========
    # lon_vals = lon.values
    # lat_vals = lat.values
    # lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')

    ds_out = xr.Dataset({
        "TuV_deg": (["lat", "lon"], TuV_deg),
        "lon": (["lat", "lon"], lon),
        "lat": (["lat", "lon"], lat),
        "pdf_values": (["x_grid"], pdf_values),
        "x_grid": (["x_grid"], x_grid),
    })

    out_path = os.path.join(out_dir, f"TuV_7d_{date_tag}.nc")
    ds_out.to_netcdf(out_path)
    print(f"Saved: {out_path}")

    # Cleanup
    ds_rho.close()

# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from glob import glob
import gsw
from numpy.linalg import lstsq
from dask import delayed, compute
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j  

# =====================
# Setup Dask cluster for parallel processing
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ===================
# Define paths and grid info
# ===================
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
rho_input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_weekly"
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle"
os.makedirs(out_dir, exist_ok=True)

# Load grid from zarr format with Dask-chunking
print("Loading grid with chunks...")
ds_grid = xr.open_zarr(zarr_path, consolidated=False, chunks={"face": 1, "i": 100, "j": 100})
depth = ds_grid.Z
lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)
dxF = ds_grid.dxF.isel(face=face, i=i, j=j)  # Zonal grid spacing
dyF = ds_grid.dyF.isel(face=face, i=i, j=j)  # Meridional grid spacing
# lon2d, lat2d = xr.broadcast(lon, lat)

# ------------------------------------------
# Define delayed function to compute horizontal gradients in tiles
# ------------------------------------------
@delayed
def process_tile(j0, j1, i0, i1, tt_np, ss_np, dxF_np, dyF_np):
    # Create output arrays filled with NaNs
    dt_dx_tile = np.full((j1 - j0, i1 - i0), np.nan)
    dt_dy_tile = np.full((j1 - j0, i1 - i0), np.nan)
    ds_dx_tile = np.full((j1 - j0, i1 - i0), np.nan)
    ds_dy_tile = np.full((j1 - j0, i1 - i0), np.nan)

    # Plane-fitting matrix for 3x3 stencil
    x = np.tile([-1, 0, 1], 3)
    y = np.repeat([-1, 0, 1], 3)
    A = np.vstack([x, y, np.ones(9)]).T

    # Loop through each valid point within the tile
    for jj, j in enumerate(range(j0, j1)):
        for ii, i in enumerate(range(i0, i1)):
            if j <= 0 or j >= tt_np.shape[0] - 1 or i <= 0 or i >= tt_np.shape[1] - 1:
                continue

            # Extract 3x3 window
            t_win = tt_np[j - 1:j + 2, i - 1:i + 2].flatten()
            s_win = ss_np[j - 1:j + 2, i - 1:i + 2].flatten()

            # Skip if any missing value
            if np.any(np.isnan(t_win)) or np.any(np.isnan(s_win)):
                continue

            # Fit plane to temperature and salinity
            coef_t, *_ = lstsq(A, t_win, rcond=None)
            coef_s, *_ = lstsq(A, s_win, rcond=None)

            # Normalize by physical grid spacing
            dt_dx_tile[jj, ii] = coef_t[0] / dxF_np[j, i]
            dt_dy_tile[jj, ii] = coef_t[1] / dyF_np[j, i]
            ds_dx_tile[jj, ii] = coef_s[0] / dxF_np[j, i]
            ds_dy_tile[jj, ii] = coef_s[1] / dyF_np[j, i]

    return dt_dx_tile, dt_dy_tile, ds_dx_tile, ds_dy_tile

# ============================
# Loop through weekly data files
# ============================
rho_files = sorted(glob(os.path.join(rho_input_dir, "rho_Hml_TS_7d_*.nc")))

for rho_file in rho_files:
    date_tag = os.path.basename(rho_file).split("_")[-1].replace(".nc", "")
    print(f"\nProcessing Turner angle for week: {date_tag}")

    # ---------------------------
    # Load weekly dataset with Dask chunks
    # ---------------------------
    ds_rho = xr.open_dataset(rho_file, chunks={"k": 10, "j": 100, "i": 100})
    rho = ds_rho["rho_7d"]
    alpha = ds_rho["alpha_7d"]
    beta = ds_rho["beta_7d"]
    Hml = ds_rho["Hml_7d"]
    T_7d = ds_rho["T_7d"]
    S_7d = ds_rho["S_7d"]

    # =============
    # Vertical Turner angle computation (TuV)
    # =============
    rho_10m = rho.isel(k=6)
    drho = rho - rho_10m.expand_dims(k=rho.k)
    drho = drho.assign_coords(k=depth)

    Hml_50 = Hml * 0.5
    Hml_90 = Hml * 0.9
    depth_vals = depth.values
    depth_3d = depth_vals[:, None, None]

    # Interpolate depth indices
    k_50 = np.abs(depth_3d - Hml_50.values[None, :, :]).argmin(axis=0)
    k_90 = np.abs(depth_3d - Hml_90.values[None, :, :]).argmin(axis=0)
    j_idx, i_idx = np.meshgrid(np.arange(k_50.shape[0]), np.arange(k_50.shape[1]), indexing='ij')

    # Extract values at those depth indices
    def extract_at_k(arr, k_idx):
        return arr.values[k_idx, j_idx, i_idx]

    tt_k50 = extract_at_k(T_7d, k_50)
    tt_k90 = extract_at_k(T_7d, k_90)
    ss_k50 = extract_at_k(S_7d, k_50)
    ss_k90 = extract_at_k(S_7d, k_90)
    alpha_k50 = extract_at_k(alpha, k_50)
    beta_k50 = extract_at_k(beta, k_50)

    dz = depth_vals[k_50] - depth_vals[k_90]
    dz = np.where(dz == 0, np.nan, dz)
    dT_dz = (tt_k50 - tt_k90) / dz
    dS_dz = (ss_k50 - ss_k90) / dz

    x_vert = beta_k50 * dS_dz
    y_vert = alpha_k50 * dT_dz
    # TuV_rad = np.arctan2((y_vert + x_vert), (y_vert - x_vert))
    TuV_rad = np.arctan((y_vert + x_vert), (y_vert - x_vert))
    TuV_deg = np.degrees(TuV_rad)

    # =============
    # Surface values (for horizontal Turner angle)
    # =============
    tt_surf = T_7d.isel(k=0)
    ss_surf = S_7d.isel(k=0)
    alpha_surf = alpha.isel(k=0)
    beta_surf = beta.isel(k=0)

    tt_np = tt_surf.values
    ss_np = ss_surf.values
    dxF_np = dxF.values
    dyF_np = dyF.values
    ny, nx = tt_np.shape

    # =============
    # Launch delayed tasks for horizontal gradient computation
    # =============
    tile_size = 100
    tasks = []
    for j0 in range(0, ny, tile_size):
        j1 = min(j0 + tile_size, ny)
        for i0 in range(0, nx, tile_size):
            i1 = min(i0 + tile_size, nx)
            task = process_tile(j0, j1, i0, i1, tt_np, ss_np, dxF_np, dyF_np)
            tasks.append((j0, j1, i0, i1, task))

    # Compute all tiles in parallel
    results = compute(*[t[4] for t in tasks])

    # Merge tile results into full fields
    dt_dx = np.full_like(tt_np, np.nan)
    dt_dy = np.full_like(tt_np, np.nan)
    ds_dx = np.full_like(tt_np, np.nan)
    ds_dy = np.full_like(tt_np, np.nan)

    for (j0, j1, i0, i1, _), (tile_dt_dx, tile_dt_dy, tile_ds_dx, tile_ds_dy) in zip(tasks, results):
        dt_dx[j0:j1, i0:i1] = tile_dt_dx
        dt_dy[j0:j1, i0:i1] = tile_dt_dy
        ds_dx[j0:j1, i0:i1] = tile_ds_dx
        ds_dy[j0:j1, i0:i1] = tile_ds_dy

    # =============
    # Horizontal Turner angle (TuH)
    # =============
    grad_rho_x = -alpha_surf * dt_dx + beta_surf * ds_dx
    grad_rho_y = -alpha_surf * dt_dy + beta_surf * ds_dy
    mag_linear = np.hypot(grad_rho_x, grad_rho_y)

    norm_x = grad_rho_x / mag_linear
    norm_y = grad_rho_y / mag_linear
    dt_cross = dt_dx * norm_x + dt_dy * norm_y
    ds_cross = ds_dx * norm_x + ds_dy * norm_y

    numerator = alpha_surf * dt_cross + beta_surf * ds_cross
    denominator = alpha_surf * dt_cross - beta_surf * ds_cross
    TuH_rad = np.arctan(numerator / denominator)
    TuH_deg = np.degrees(TuH_rad)

    # =============
    # Differences and KDE PDFs
    # =============
    Tu_diff = np.abs(TuV_deg - TuH_deg)

    TuV_clean = TuV_deg.ravel()[~np.isnan(TuV_deg.ravel())]
    TuH_clean = TuH_deg.data.ravel()[~np.isnan(TuH_deg.data.ravel())]

    if TuV_clean.size < 100 or TuH_clean.size < 100:
        print(f"Insufficient data for KDE, skipping week: {date_tag}")
        ds_rho.close()
        continue

    x_grid = np.linspace(-180, 180, 1000)
    pdf_v = gaussian_kde(TuV_clean, bw_method=0.05)(x_grid)
    pdf_h = gaussian_kde(TuH_clean, bw_method=0.05)(x_grid)

    # =============
    # Save to NetCDF
    # =============
    ds_out = xr.Dataset(
        {
            "TuV_deg": (["j", "i"], TuV_deg),
            "TuH_deg": (["j", "i"], TuH_deg.data),
            "Tu_diff": (["j", "i"], Tu_diff.data),
            "alpha_surf": (["j", "i"], alpha_surf.data),
            "beta_surf": (["j", "i"], beta_surf.data),
            "lon": (["i", "j"], lon.data),
            "lat": (["i", "j"], lat.data),
            "pdf_values_v": (["x_grid"], pdf_v),
            "pdf_values_h": (["x_grid"], pdf_h),
        },
        coords={
            "j": TuH_deg.coords["j"],
            "i": TuH_deg.coords["i"],
            "x_grid": x_grid,
            "face": TuH_deg.coords["face"],
            "k": TuH_deg.coords["k"],
        },
    )

    out_path = os.path.join(out_dir, f"TuVH_7d_{date_tag}.nc")
    ds_out.to_netcdf(out_path)
    print(f"Saved: {out_path}")
    ds_rho.close()

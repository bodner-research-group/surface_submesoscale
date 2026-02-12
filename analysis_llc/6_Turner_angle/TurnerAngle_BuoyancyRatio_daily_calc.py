### Compute Turner angles and buoyancy ratio r using daily averaged temperature and salinity

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

# ============================================================
#                       DOMAIN
# ============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

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
rho_input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
# out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_BuoyancyRatio_daily"
out_dir = f"/home/y_si/orcd/scratch/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_BuoyancyRatio_daily"
os.makedirs(out_dir, exist_ok=True)


# Load grid from zarr format with Dask-chunking
print("Loading grid with chunks...")
ds_grid = xr.open_zarr(zarr_path, consolidated=False, chunks={"face": 1, "i": 100, "j": 100})
depth = ds_grid.Z
lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)
dxF = ds_grid.dxF.isel(face=face, i=i, j=j)  # Zonal grid spacing
dyF = ds_grid.dyF.isel(face=face, i=i, j=j)  # Meridional grid spacing


from xgcm import Grid
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
# ========= Load grid data =========
# print("Loading grid...")
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

# ========= Compute derivatives =========
def compute_grad(var, grid):
    
    # First derivatives on grid edges
    var_x = grid.derivative(var, axis="X")  # ∂var/∂x
    var_y = grid.derivative(var, axis="Y")  # ∂var/∂y
    # Interpolate first derivatives back to cell centers for gradient magnitude
    var_x_c = grid.interp(var_x, axis="X", to="center")
    var_y_c = grid.interp(var_y, axis="Y", to="center")
    grad_mag = np.sqrt(var_x_c**2 + var_y_c**2)
    grad_mag = grad_mag.assign_coords(time=var.time)

    return grad_mag



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
            if np.any(np.isnan(t_win)) or np.any(np.isnan(s_win)) :
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
# Loop through daily data files
# ============================
rho_files = sorted(glob(os.path.join(rho_input_dir, "rho_Hml_TS_daily_*.nc")))


# for rho_file in rho_files:
for rho_file in rho_files[::-1]:
    # #### test one day
    # rho_file = os.path.join(rho_input_dir, "rho_Hml_TS_daily_20121029.nc")
    # ####

    date_tag = os.path.basename(rho_file).split("_")[-1].replace(".nc", "")
    out_path = os.path.join(
        out_dir,
        f"TurnerAngle_BuoyancyRatio_daily_{date_tag}.nc"
    )

    # 🔹 if file exsits, skip it
    if os.path.exists(out_path):
        print(f"Skipping {date_tag} (already exists)")
        continue

    print(f"\nProcessing Turner angle for day: {date_tag}")

    # ---------------------------
    # Load weekly dataset with Dask chunks
    # ---------------------------
    ds_rho = xr.open_dataset(rho_file, chunks={"k": 10, "j": 100, "i": 100})
    rho = ds_rho["rho_daily"]
    alpha = ds_rho["alpha_daily"]
    beta = ds_rho["beta_daily"]
    T_daily = ds_rho["T_daily"]
    S_daily = ds_rho["S_daily"]

    Hml_file = os.path.join(rho_input_dir, f"Hml_daily_surface_reference_{date_tag}.nc") 
    ds_Hml = xr.open_dataset(Hml_file, chunks={"j": 100, "i": 100})
    Hml = ds_Hml["Hml_daily"]

    # =============
    # Vertical Turner angle computation (TuV)
    # =============
    Hml_50 = Hml * 0.5
    Hml_70 = Hml * 0.7
    Hml_90 = Hml * 0.9
    depth_vals = depth.values
    depth_3d = depth_vals[:, None, None]

    # Interpolate depth indices
    k_50 = np.abs(depth_3d - Hml_50.values[None, :, :]).argmin(axis=0)
    k_70 = np.abs(depth_3d - Hml_70.values[None, :, :]).argmin(axis=0)
    k_90 = np.abs(depth_3d - Hml_90.values[None, :, :]).argmin(axis=0)
    j_idx, i_idx = np.meshgrid(np.arange(k_50.shape[0]), np.arange(k_50.shape[1]), indexing='ij')

    # Extract values at those depth indices
    def extract_at_k(arr, k_idx):
        return arr.values[k_idx, j_idx, i_idx]

    tt_k50 = extract_at_k(T_daily, k_50)
    tt_k90 = extract_at_k(T_daily, k_90)
    ss_k50 = extract_at_k(S_daily, k_50)
    ss_k90 = extract_at_k(S_daily, k_90)

    alpha_k50 = extract_at_k(alpha, k_50)
    alpha_k70 = extract_at_k(alpha, k_70)
    alpha_k90 = extract_at_k(alpha, k_90)

    beta_k70 = extract_at_k(beta, k_70)


    dz = depth_vals[k_50] - depth_vals[k_90]
    dz = np.where(dz == 0, np.nan, dz)
    dT_dz = (tt_k50 - tt_k90) / dz
    dS_dz = (ss_k50 - ss_k90) / dz

    x_vert = beta_k70 * dS_dz
    y_vert = alpha_k70 * dT_dz
    # TuV_rad = np.arctan2((y_vert + x_vert), (y_vert - x_vert))
    numerator_V = y_vert + x_vert
    denominator_V = y_vert - x_vert
    TuV_rad = np.arctan(numerator_V / denominator_V)
    TuV_deg = np.degrees(TuV_rad)

    # =============
    # Surface values (for horizontal Turner angle)
    # =============
    tt_np = T_daily.isel(k=0).values
    ss_np = S_daily.isel(k=0).values
    alpha_surf = alpha.isel(k=0)
    beta_surf = beta.isel(k=0)

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
    ds_dx = np.full_like(ss_np, np.nan)
    ds_dy = np.full_like(ss_np, np.nan)


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
    # The buoyancy ratio r
    # =============
    buoyancy_ratio_linearEOS = alpha_surf * dt_cross / denominator

    rho_0 = 1027.5
    rho_s = rho.isel(k=0)
    rho_grad_mag = compute_grad(rho_s, grid)
    buoyancy_ratio = alpha_surf * dt_cross / (-rho_grad_mag/rho_0)

    # =============
    # Differences and KDE PDFs
    # =============
    Tu_diff = np.abs(TuV_deg - TuH_deg)

    TuV_clean = TuV_deg.ravel()[~np.isnan(TuV_deg.ravel())]
    TuH_clean = TuH_deg.data.ravel()[~np.isnan(TuH_deg.data.ravel())]


    x_grid = np.linspace(-180, 180, 361)  # 1-degree bins
    pdf_v = gaussian_kde(TuV_clean, bw_method=0.05)(x_grid)
    pdf_h = gaussian_kde(TuH_clean, bw_method=0.05)(x_grid)

    # =============
    # Save to NetCDF
    # =============
    ds_out = xr.Dataset(
        {
            "dt_cross": (["j", "i"], dt_cross.data),
            "ds_cross": (["j", "i"], ds_cross.data),
            "TuV_deg": (["j", "i"], TuV_deg),
            "TuH_deg": (["j", "i"], TuH_deg.data),
            "buoyancy_ratio_linearEOS": (["j", "i"], buoyancy_ratio_linearEOS.data),
            "buoyancy_ratio": (["j", "i"], buoyancy_ratio.data),
            "Tu_diff": (["j", "i"], Tu_diff.data),
            "alpha_surf": (["j", "i"], alpha_surf.data),
            "beta_surf": (["j", "i"], beta_surf.data),
            # "lon": (["i", "j"], lon.data),
            # "lat": (["i", "j"], lat.data),
            # "lon": lon,
            # "lat": lat,
            "pdf_values_v": (["x_grid"], pdf_v),
            "pdf_values_h": (["x_grid"], pdf_h),
            "alpha_k50": (["j", "i"], alpha_k50),
            "alpha_k70": (["j", "i"], alpha_k70),
            "alpha_k90": (["j", "i"], alpha_k90),
            "beta_k70": (["j", "i"], beta_k70),
        },
        coords={
            "j": TuH_deg.coords["j"],
            "i": TuH_deg.coords["i"],
            "x_grid": x_grid,
            "face": TuH_deg.coords["face"],
            "k": TuH_deg.coords["k"],
        },
    )

    # ds_out.to_netcdf(out_path)

    ds_out = ds_out.compute()  
    ds_out.to_netcdf(out_path)
    print(f"Saved: {out_path}")

    ds_rho.close()
    ds_out.close()  




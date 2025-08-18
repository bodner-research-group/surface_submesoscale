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
import dask.array as da
import scipy.signal

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
lat = ds_grid.YC.isel(face=face, i=1, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=1)
dxF = ds_grid.dxF.isel(face=face, i=i, j=j)  # Zonal grid spacing
dyF = ds_grid.dyF.isel(face=face, i=i, j=j)  # Meridional grid spacing
lon2d, lat2d = xr.broadcast(lon, lat)

# =============================
# Prepare 3x3 least squares weights for gradient calculation
# =============================
# Construct design matrix A for 3x3 stencil points (x,y,1)
x_coords = np.tile([-1, 0, 1], 3)
y_coords = np.repeat([-1, 0, 1], 3)
A = np.vstack([x_coords, y_coords, np.ones(9)]).T  # Shape (9,3)

# Compute pseudoinverse of A to get weights for dx and dy
W = np.linalg.pinv(A)  # Shape (3,9)
wx = W[0].reshape(3, 3)  # Weights for gradient in x-direction
wy = W[1].reshape(3, 3)  # Weights for gradient in y-direction

# =============================
# Function to compute gradients with Dask and convolution
# =============================
def compute_gradient_dask(field, wx, wy, dxF, dyF):
    """
    Compute horizontal gradients of a 2D field using 3x3 least-squares fitting,
    applied with dask.array.map_overlap for parallel convolution with overlapping chunks.

    Parameters:
        field: dask.array of 2D scalar field (e.g. temperature or salinity)
        wx, wy: 3x3 convolution kernels for x and y gradient respectively
        dxF, dyF: dask.array grid spacing in x and y directions

    Returns:
        dt_dx, dt_dy: dask.array of gradients normalized by grid spacing
    """
    def convolve_block(block):
        # block shape includes overlap padding (chunk_size+2, chunk_size+2)
        grad_x = scipy.signal.convolve2d(block, wx, mode='valid', boundary='symm')
        grad_y = scipy.signal.convolve2d(block, wy, mode='valid', boundary='symm')
        return np.stack([grad_x, grad_y])

    # Use map_overlap with depth=1 (1 pixel overlap) to handle 3x3 stencil at chunk edges
    stacked = da.map_overlap(convolve_block, field,
                            depth=1, boundary='reflect',
                            trim=True, dtype=float)

    # stacked shape: (2, height, width) -> split into dx, dy
    grad_dx = stacked[0]
    grad_dy = stacked[1]

    # Normalize gradients by grid spacing (element-wise division)
    grad_dx = grad_dx / dxF
    grad_dy = grad_dy / dyF

    return grad_dx, grad_dy


# ============================
# Loop through weekly data files
# ============================
rho_files = sorted(glob(os.path.join(rho_input_dir, "rho_Hml_TS_7d_*.nc")))

for rho_file in rho_files[0:10]:
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
    # Vertical Turner angle calculation (TuV)
    # =============
    rho_10m = rho.isel(k=6)
    drho = rho - rho_10m.expand_dims(k=rho.k)
    drho = drho.assign_coords(k=depth)

    Hml_50 = Hml * 0.5
    Hml_90 = Hml * 0.9
    depth_vals = depth.values
    depth_3d = depth_vals[:, None, None]

    # Find indices closest to half and 90% mixed layer depth
    k_50 = np.abs(depth_3d - Hml_50.values[None, :, :]).argmin(axis=0)
    k_90 = np.abs(depth_3d - Hml_90.values[None, :, :]).argmin(axis=0)
    j_idx, i_idx = np.meshgrid(np.arange(k_50.shape[0]), np.arange(k_50.shape[1]), indexing='ij')

    def extract_at_k(arr, k_idx):
        # Extract 2D slice of arr at varying vertical indices k_idx[j,i]
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
    TuV_rad = np.arctan2((y_vert + x_vert), (y_vert - x_vert))
    TuV_deg = np.degrees(TuV_rad)

    # =============
    # Surface values for horizontal Turner angle (TuH)
    # =============
    tt_surf = T_7d.isel(k=0)
    ss_surf = S_7d.isel(k=0)
    alpha_surf = alpha.isel(k=0)
    beta_surf = beta.isel(k=0)

    # Convert surface fields and grid spacing to dask arrays
    tt_dask = tt_surf.chunk({"j": -1, "i": -1}).data
    ss_dask = ss_surf.chunk({"j": -1, "i": -1}).data
    dxF_dask = da.from_array(dxF.values, chunks=dxF.values.shape)
    dyF_dask = da.from_array(dyF.values, chunks=dyF.values.shape)

    # =============
    # Compute horizontal gradients using Dask convolution function
    # =============
    dt_dx, dt_dy = compute_gradient_dask(tt_dask, wx, wy, dxF_dask, dyF_dask)
    ds_dx, ds_dy = compute_gradient_dask(ss_dask, wx, wy, dxF_dask, dyF_dask)

    # Compute to numpy arrays
    dt_dx_np = dt_dx.compute()
    dt_dy_np = dt_dy.compute()
    ds_dx_np = ds_dx.compute()
    ds_dy_np = ds_dy.compute()

    # =============
    # Horizontal Turner angle calculation (TuH)
    # =============
    # alpha_surf, beta_surf are xarray DataArrays; convert to numpy for arithmetic
    alpha_np = alpha_surf.values
    beta_np = beta_surf.values

    grad_rho_x = -alpha_np * dt_dx_np + beta_np * ds_dx_np
    grad_rho_y = -alpha_np * dt_dy_np + beta_np * ds_dy_np
    mag_linear = np.hypot(grad_rho_x, grad_rho_y)

    # Avoid division by zero
    mag_linear_safe = np.where(mag_linear == 0, np.nan, mag_linear)

    norm_x = grad_rho_x / mag_linear_safe
    norm_y = grad_rho_y / mag_linear_safe
    dt_cross = dt_dx_np * norm_x + dt_dy_np * norm_y
    ds_cross = ds_dx_np * norm_x + ds_dy_np * norm_y

    numerator = alpha_np * dt_cross + beta_np * ds_cross
    denominator = alpha_np * dt_cross - beta_np * ds_cross
    TuH_rad = np.arctan(numerator / denominator)
    TuH_deg = np.degrees(TuH_rad)

    # =============
    # Differences and KDE PDFs
    # =============
    Tu_diff = np.abs(TuV_deg - TuH_deg)

    TuV_clean = TuV_deg.ravel()[~np.isnan(TuV_deg.ravel())]
    TuH_clean = TuH_deg.ravel()[~np.isnan(TuH_deg.ravel())]

    if TuV_clean.size < 100 or TuH_clean.size < 100:
        print(f"Insufficient data for KDE, skipping week: {date_tag}")
        ds_rho.close()
        continue

    x_grid = np.linspace(-180, 180, 1000)
    kde_V = gaussian_kde(TuV_clean, bw_method=0.3)
    kde_H = gaussian_kde(TuH_clean, bw_method=0.3)
    kde_V_vals = kde_V(x_grid)
    kde_H_vals = kde_H(x_grid)

    # =============
    # Save results to netCDF
    # =============
    out_file = os.path.join(out_dir, f"TurnerAngle_{date_tag}.nc")
    ds_out = xr.Dataset(
        data_vars=dict(
            TuV_deg=(["j", "i"], TuV_deg),
            TuH_deg=(["j", "i"], TuH_deg),
            Tu_diff=(["j", "i"], Tu_diff),
            kde_x=(["kde_pts"], x_grid),
            kde_V=(["kde_pts"], kde_V_vals),
            kde_H=(["kde_pts"], kde_H_vals),
        ),
        coords=dict(
            i=i,
            j=j,
        ),
        attrs=dict(
            description=f"Turner Angle vertical and horizontal comparison for week {date_tag}"
        )
    )
    ds_out.to_netcdf(out_file)
    print(f"Saved output: {out_file}")

    ds_rho.close()

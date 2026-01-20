###############################################################
#  Compute hourly mixed-layer eddy buoyancy flux wb_eddy
#  using Gaussian filtering (SEQUENTIAL VERSION)
###############################################################

import xarray as xr
import numpy as np
import os
import gc
from glob import glob
from scipy.ndimage import gaussian_filter
from dask.distributed import Client, LocalCluster

# ============================================================
#                       DOMAIN
# ============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ============================================================
#                       CONSTANTS
# ============================================================
gravity = 9.81
rho0 = 1027.5
min_H = 10.0

# ============================================================
#                       PATHS
# ============================================================
rho_dir = (
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/"
    f"analysis_llc/data/{domain_name}/hourly_rho_Hml"
)

out_dir = (
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/"
    f"analysis_llc/data/{domain_name}/hourly_wb_eddy_gaussian_wide"
)
os.makedirs(out_dir, exist_ok=True)

Lambda_file = (
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/"
    f"analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily_surface_reference.nc"
)

# ============================================================
#                 MIXED-LAYER INTEGRAL
# ============================================================
def ml_integrated_profile(var, Hml, depth, dz, min_H):
    if Hml < min_H:
        return np.nan
    mask = depth >= -Hml
    if not np.any(mask):
        return np.nan
    num = np.sum(var[mask] * dz[mask])
    den = np.sum(dz[mask])
    return np.nan if den == 0 else num / den


def ml_integral(var, Hml, depth, dz, min_H):
    return xr.apply_ufunc(
        ml_integrated_profile,
        var,
        Hml,
        depth,
        dz,
        input_core_dims=[["k"], [], ["k"], ["k"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        kwargs={"min_H": min_H},
        output_dtypes=[float],
    )

# ============================================================
#                           MAIN
# ============================================================
# def main():

# ================= DASK (for xarray ops only) =================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ================= GRID =================
ds_grid = xr.open_zarr(
    "/orcd/data/abodner/003/LLC4320/LLC4320",
    consolidated=False,
)

depth = ds_grid.Z
depth_kp1 = ds_grid.Zp1
dz = ds_grid.drF

lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)
dz3d, _, _ = xr.broadcast(dz, lon, lat)

# ================= GRID SPACING =================
dxC = ds_grid.dxC.isel(face=face, i_g=i, j=j).values.mean()
dyC = ds_grid.dyC.isel(face=face, i=i, j_g=j).values.mean()
dx_km = np.sqrt(0.5 * (dxC**2 + dyC**2)) / 1000.0
print(f"Grid spacing ≈ {dx_km:.2f} km")

# ================= W =================
ndays = 366
start_hours = 49 * 24
end_hours = start_hours + 24 * ndays
time = slice(start_hours, end_hours)

W_all = ds_grid.W.isel(face=face, i=i, j=j, time=time)
W_all = W_all.chunk({"time": 1, "j": -1, "i": -1})

# ================= Lambda_MLI (time-mean) =================
ds_lambda = xr.open_dataset(Lambda_file)
lambda_window = ds_lambda.Lambda_MLI_mean.isel(time=slice(61, 61 + 62))
lambda_km = float(lambda_window.mean().values) / 1000.0

# sigma_km = lambda_km / np.sqrt(8.0 * np.log(2.0))
sigma_km = lambda_km
sigma_pts = sigma_km / dx_km

print(
    f"Gaussian filter: Λ̄_MLI = {lambda_km:.2f} km "
    f"→ σ = {sigma_pts:.2f} grid pts"
)

# ================= FILE LIST =================
rho_files = sorted(glob(os.path.join(rho_dir, "rho_Hml_*.nc")))
target_files = rho_files[0 * 24 : 366 * 24 + 1]

# ============================================================
#                   MAIN LOOP (SEQUENTIAL)
# ============================================================
for f in target_files:

    tag = os.path.basename(f).replace("rho_Hml_", "").replace(".nc", "")
    out_file = os.path.join(out_dir, f"wb_eddy_{tag}.nc")

    if os.path.exists(out_file):
        print(f"Skipping {tag}")
        continue

    print(f"Processing {tag}")

    ds_rho = xr.open_dataset(f)

    rho = ds_rho["rho"].chunk({"j": -1, "i": -1})
    Hml = -ds_rho["Hml_SurfRef"].chunk({"j": -1, "i": -1})

    # ================= MATCH W TIME =================
    t_index = int(np.argmin(np.abs(W_all.time.values - rho.time.values)))
    w_kp1 = W_all.isel(time=t_index)

    w_k = (
        w_kp1.rename({"k_p1": "k"})
                .assign_coords(k=depth_kp1.values)
                .interp(k=depth)
                .fillna(0)
    )

    # ================= BUOYANCY =================
    b = -gravity * (rho - rho0) / rho0

    # ================= GAUSSIAN FILTER =================
    def gfilter(x):
        return gaussian_filter(
            x,
            sigma=(0, sigma_pts, sigma_pts),
            mode="reflect",
        )

    w_f = xr.apply_ufunc(gfilter, w_k, dask="parallelized", output_dtypes=[float])
    b_f = xr.apply_ufunc(gfilter, b,   dask="parallelized", output_dtypes=[float])
    # wb_f = xr.apply_ufunc(gfilter, w_k * b, dask="parallelized", output_dtypes=[float])

    # ================= MIXED-LAYER AVERAGES =================
    # wb_total = ml_integral(wb_f,        Hml, depth, dz3d, min_H)
    # wb_mean  = ml_integral(w_f * b_f,   Hml, depth, dz3d, min_H)
    # wb_eddy   = wb_total - wb_mean

    wb_eddy   = ml_integral((w_k-w_f)*(b-b_f), Hml, depth, dz3d, min_H)
    wb_total  = ml_integral(w_k * b,           Hml, depth, dz3d, min_H)
    wb_mean   = wb_total - wb_eddy
    
    # ================= SAVE =================
    ds_out = xr.Dataset(
        {
            "wb_total": wb_total,
            "wb_mean": wb_mean,
            "wb_eddy": wb_eddy,
            # "Hml": Hml,
        },
        coords={"time": rho.time},
    )

    ds_out.to_netcdf(out_file)
    print(f"Saved → {out_file}")

    ds_rho.close()
    del ds_out, rho, Hml, w_k
    gc.collect()

# ================= CLEANUP =================
ds_grid.close()
ds_lambda.close()
client.close()
cluster.close()


# ============================================================
#                       ENTRY POINT
# ============================================================
# if __name__ == "__main__":
#     main()

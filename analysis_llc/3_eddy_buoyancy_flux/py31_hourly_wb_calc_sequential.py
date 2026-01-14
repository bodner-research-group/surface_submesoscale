###############################################################
#  Compute hourly mixed-layer eddy buoyancy flux wb_eddy
#  using hourly rho/Hml and model W
###############################################################

import xarray as xr
import numpy as np
import os
import gc
from glob import glob
from dask.distributed import Client, LocalCluster

# ============================================================
#                       DOMAIN
# ============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# def main():
# ============================================================
#                       DASK
# ============================================================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ============================================================
#                       PATHS
# ============================================================
rho_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/hourly_rho_Hml"
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/hourly_wb_eddy_window14"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
#                     LOAD MODEL GRID
# ============================================================
ds_grid = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

lat   = ds_grid.YC.isel(face=face, i=i, j=j)
lon   = ds_grid.XC.isel(face=face, i=i, j=j)
depth = ds_grid.Z        # (k)
depth_kp1 = ds_grid.Zp1  # (k+1)
dz = ds_grid.drF

dz3d, _, _ = xr.broadcast(dz, lon, lat)

# ============================================================
#                 HOURLY VERTICAL VELOCITY
# ============================================================
####### Important!! Don't change the time indices.
ndays = 366
start_hours = 49 * 24
end_hours   = start_hours + 24 * ndays
time = slice(start_hours, end_hours)

W_all = ds_grid.W.isel(face=face, i=i, j=j, time=time)
W_all = W_all.chunk({"time": 1, "j": -1, "i": -1})

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
#                       CONSTANTS
# ============================================================
gravity = 9.81
rho0 = 1027.5
min_H = 10.0
window = 14  # 14/48 degree

# ============================================================
#                     FILE LIST
# ============================================================
rho_files = sorted(glob(os.path.join(rho_dir, "rho_Hml_*.nc")))

# ============================================================
#                     MAIN LOOP
# ============================================================
# for f in rho_files[180*24-1 : 120*24-1 : -1]:
for f in rho_files:

    tag = os.path.basename(f).replace("rho_Hml_", "").replace(".nc", "")
    out_file = os.path.join(out_dir, f"wb_eddy_{tag}.nc")

    if os.path.exists(out_file):
        print(f"Skipping {tag}")
        continue

    print(f"Processing {tag}")

    ds_rho = xr.open_dataset(f)

    rho = ds_rho["rho"].chunk({"j": -1, "i": -1})
    Hml = ds_rho["Hml_SurfRef"].chunk({"j": -1, "i": -1})
    Hml = -Hml  # make positive depth

    # match model W time
    t_index = int(np.argmin(np.abs(W_all.time.values - rho.time.values)))
    w_kp1 = W_all.isel(time=t_index)

    # interpolate W(k+1) → W(k)
    w_k = (
        w_kp1.rename({"k_p1": "k"})
                .assign_coords(k=depth_kp1.values)
                .interp(k=depth)
                .fillna(0)
    )

    # buoyancy
    b = -gravity * (rho - rho0) / rho0

    # ========================================================
    #                  COARSE GRAINING
    # ========================================================
    Hml_cg = Hml.coarsen(i=window, j=window, boundary="trim").mean()
    b_cg   = b.coarsen(i=window, j=window, boundary="trim").mean()
    w_cg   = w_k.coarsen(i=window, j=window, boundary="trim").mean()
    dz_cg  = dz3d.coarsen(i=window, j=window, boundary="trim").mean()

    wb_cg = (w_k * b).coarsen(i=window, j=window, boundary="trim").mean()

    # ========================================================
    #              MIXED-LAYER AVERAGES
    # ========================================================
    wb_avg  = ml_integral(wb_cg,       Hml_cg, depth, dz_cg, min_H)
    wb_fact = ml_integral(w_cg * b_cg, Hml_cg, depth, dz_cg, min_H)

    B_eddy = wb_avg - wb_fact

    # ========================================================
    #                     SAVE
    # ========================================================
    ds_out = xr.Dataset(
        {
            "wb_avg": wb_avg,
            "wb_fact": wb_fact,
            "B_eddy": B_eddy,
            "Hml": Hml_cg,
        },
        coords={"time": rho.time},
    )

    ds_out.to_netcdf(out_file)
    print(f"Saved → {out_file}")

    ds_rho.close()
    del ds_out, rho, Hml, w_k
    gc.collect()

ds_grid.close()
client.close()
cluster.close()


# # ========== Entry Point ==========
# if __name__ == "__main__":
# main()

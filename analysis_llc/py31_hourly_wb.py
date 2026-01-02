### Compute mixed-layer depth-averages of coarse-grained w, b, wb 
### Exclude regions where |Hml| <10m

import xarray as xr
import numpy as np
import os
from glob import glob
from dask.distributed import Client, LocalCluster

# from set_constant import domain_name, face, i, j

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ============================================================
#                        DASK
# ============================================================
cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit="11GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ============================================================
#                      DIRECTORIES
# ============================================================
rho_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_hourly"
w_dir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_hourly"
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_hourly_1_12deg"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
#                    LOAD GRID INFO
# ============================================================
ds_grid = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)

depth = ds_grid.Z          # (k)
depth_kp1 = ds_grid.Zp1    # (k+1)
dz = ds_grid.drF           # (k)
# broadcast dz to (k,j,i)
dz3d, _, _ = xr.broadcast(dz, lon, lat)


# ============================================================
#                      FILE LISTS
# ============================================================
rho_files = sorted(glob(os.path.join(rho_dir, "rho_Hml_*.nc")))
hml_files = rho_files

ww_files  = sorted(glob(os.path.join(w_dir,  "ww_24h_*.nc")))
min_H = 10.0

# ============================================================
#                        Load W data
# ============================================================
print("Loading all weekly W files into one dataset...")
ww_list = []
for f in ww_files:
    ds = xr.open_dataset(f)["W"]
    # chunk time=1 to keep interpolation cheap
    ds = ds.chunk({"time": 1, "i": -1, "j": -1})
    ww_list.append(ds)
# Concatenate into one long time series
W_all = xr.concat(ww_list, dim="time").sortby("time")
W_times = W_all.time.values
print(f"W_all loaded: shape={W_all.shape}, ntimes={len(W_times)}")


# ============================================================
#      SCALAR COLUMN-WISE INTEGRATION (1D numpy arrays)
# ============================================================
def ml_integrated_profile(var_profile, Hml_value, depth_profile, dz_profile, min_H):
    """
    Mixed-layer depth-average for one vertical profile.

    Parameters:
        var_profile: 1D numpy array (k)
        Hml_value:   scalar
        depth_profile: 1D numpy array (k) negative downward
        dz_profile: 1D numpy array (k)
    """
    # Shallow HML → NaN
    if Hml_value < min_H:
        return np.nan

    # depth = negative downward, so include 0 → -Hml
    mask = depth_profile >= -Hml_value

    if not np.any(mask):
        return np.nan

    num = np.sum(var_profile[mask] * dz_profile[mask])
    den = np.sum(dz_profile[mask])

    if den == 0:
        return np.nan

    return num / den

# ============================================================
#    XARRAY WRAPPER USING apply_ufunc FOR (k,j,i) ARRAYS
# ============================================================
def ml_integral(var, Hml, depth, dz, min_H):
    """
    Compute MLD depth-average of var(k,j,i).
    """
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
        kwargs={"min_H": min_H}
    )


# ============================================================
#                          MAIN LOOP
# ============================================================

# Build dicts indexed by date_tag
rho_dict = {os.path.basename(f).split("_")[-1].replace(".nc", ""): f
            for f in rho_files}

hml_dict = {os.path.basename(f).split("_")[-1].replace(".nc", ""): f
            for f in hml_files}

# Loop over dates that exist in both
for date_tag in sorted(rho_dict.keys())[61:61+91]:
    if date_tag not in hml_dict:
        print(f"WARNING: missing Hml file for {date_tag}")
        continue

    rho_file = rho_dict[date_tag]
    Hml_file = hml_dict[date_tag]

    out_file = os.path.join(output_dir, f"wb_mld_daily_{date_tag}.nc")

    if os.path.exists(out_file):
        print(f"Skipping {date_tag} (exists)")
        continue

    print(f"Processing {date_tag} ...")

    # load daily rho & Hml
    rho = xr.open_dataset(rho_file)["rho_daily"].chunk({"i": -1, "j": -1})
    Hml = -xr.open_dataset(Hml_file)["Hml_daily"].chunk({"i": -1, "j": -1})
    rho_time = rho.time.values

    # find nearest W snapshot across ALL W times
    t_index = int(np.argmin(np.abs(W_times - rho_time)))

    print(f"ρ date = {rho_time}, using W time = {W_times[t_index]}")

    w_kp1 = W_all.isel(time=t_index)

    gravity = 9.81
    rho0 = 1027.5

    # buoyancy
    b = -gravity * (rho - rho0) / rho0  # (k,j,i)

    # interpolate W(k+1) → W(k)
    w_k = (
        w_kp1.rename({"k_p1": "k"})
                .assign_coords(k=depth_kp1.values)
                .interp(k=depth)
                .fillna(0)
    )

    # ------------------------------------------------------------------
    # ⬇ COARSE-GRAINING APPLIED HERE BEFORE VERTICAL INTEGRAL ⬇
    # ------------------------------------------------------------------
    window_size=4

    Hml_cg = Hml.coarsen(i=window_size, j=window_size, boundary="trim").mean()
    b_cg  = b.coarsen(i=window_size, j=window_size, boundary="trim").mean()
    w_cg  = w_k.coarsen(i=window_size, j=window_size, boundary="trim").mean()
    wb_cg = (w_k * b).coarsen(i=window_size, j=window_size, boundary="trim").mean()
    dz_cg = dz3d.coarsen(i=window_size, j=window_size, boundary="trim").mean()

    # ------------------------------------------------------------------
    # Vertical MLD averages
    # ------------------------------------------------------------------
    wb_avg = ml_integral(wb_cg, Hml_cg, depth, dz_cg, min_H)
    # b_avg  = ml_integral(b_cg,  Hml_cg, depth, dz_cg, min_H)
    # w_avg  = ml_integral(w_cg,  Hml_cg, depth, dz_cg, min_H)

    # separated product

    ####### version1: \overline{w'b'}^{xyz} = \overline{wb}^{xyz} - \overline{w}^{xyz} * \overline{b}^{xyz}
    # wb_fact = w_avg * b_avg

    ####### version2 -- correct: \overline{w'b'}^{xyz} = \overline{wb}^{xyz} - \overline{ \overline{w}^{xy} * \overline{b}^{xy} }^z
    wb_fact = ml_integral(w_cg*b_cg,  Hml_cg, depth, dz_cg, min_H)

    # Eddy buoyancy flux
    B_eddy = wb_avg - wb_fact

    # # --------------------------------------------------------
    # # Eddy flux at surface (k = 0)
    # # --------------------------------------------------------
    # wb_eddy_surf = wb_cg.isel(k=0) - w_cg.isel(k=0) * b_cg.isel(k=0)

    # # --------------------------------------------------------
    # # Eddy flux at local mixed-layer base (Dask-safe)
    # # --------------------------------------------------------
    # z3d, H3d = xr.broadcast(depth, Hml_cg)
    # ml_mask = z3d >= -H3d

    # # mask values outside mixed layer
    # wb_masked = wb_cg.where(ml_mask)
    # w_masked  = w_cg.where(ml_mask)
    # b_masked  = b_cg.where(ml_mask)

    # # take deepest valid k
    # wb_mlb = wb_masked.isel(k=slice(None, None, -1)).max("k")
    # w_mlb  = w_masked.isel(k=slice(None, None, -1)).max("k")
    # b_mlb  = b_masked.isel(k=slice(None, None, -1)).max("k")

    # wb_eddy_mlb = wb_mlb - w_mlb * b_mlb

    ds_out = xr.Dataset({
        "wb_avg":         wb_avg,
        "wb_fact":        wb_fact,
        "B_eddy":         B_eddy,
        "Hml_cg":         Hml_cg,
        # "w_avg":          w_avg,
        # "b_avg":          b_avg,
        # "wb_eddy_surf":   wb_eddy_surf,
        # "wb_eddy_mlb":    wb_eddy_mlb,
    })

    ds_out.to_netcdf(out_file)
    print(f"Saved → {out_file}")

    del rho, Hml, ds_out, w_k, w_kp1
    import gc
    gc.collect()


client.close()
cluster.close()

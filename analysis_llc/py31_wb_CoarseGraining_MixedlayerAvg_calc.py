### Compute mixed-layer depth-averages of coarse-grained w, b, wb 
### Exclude regions where |Hml| <10m

import xarray as xr
import numpy as np
import os
from glob import glob
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j


# ============================================================
#                      DIRECTORIES
# ============================================================
rho_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
w_dir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"

output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_daily_1_4deg"

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
rho_files = sorted(glob(os.path.join(rho_dir, "rho_Hml_TS_daily_*.nc")))
hml_files = sorted(glob(os.path.join(rho_dir, "Hml_daily_surface_reference_*.nc")))
ww_files  = sorted(glob(os.path.join(w_dir,  "ww_24h_*.nc")))


# ============================================================
#      SCALAR COLUMN-WISE INTEGRATION (1D numpy arrays)
# ============================================================
def ml_integrated_profile(var_profile, Hml_value, depth_profile, dz_profile, min_H=10.0):
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
def ml_integral(var, Hml, depth, dz, min_H=10.0):
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
#      COMPUTE MLD INTEGRALS FOR ONE TIME SNAPHOT
# ============================================================
def compute_mld_integrals_one_time(rho, Hml, w_kp1):

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
    Hml_cg = coarse_grain(Hml)
    b_cg  = coarse_grain(b)
    w_cg  = coarse_grain(w_k)
    wb_cg = coarse_grain(w_k * b)

    # coarse-grain dz (only depends on k → broadcast later)
    # dz_cg = dz3d.isel(j=slice(0, None, 4), i=slice(0, None, 4))
    dz_cg = coarse_grain(dz3d)

    # ------------------------------------------------------------------
    # Vertical MLD averages
    # ------------------------------------------------------------------
    wb_avg = ml_integral(wb_cg, Hml_cg, depth, dz_cg, min_H=10.0)
    b_avg  = ml_integral(b_cg,  Hml_cg, depth, dz_cg, min_H=10.0)
    w_avg  = ml_integral(w_cg,  Hml_cg, depth, dz_cg, min_H=10.0)

    # separated product

    ####### version1: \overline{w'b'}^{xyz} = \overline{wb}^{xyz} - \overline{w}^{xyz} * \overline{b}^{xyz}
    # wb_fact = w_avg * b_avg

    ####### version2 -- correct: \overline{w'b'}^{xyz} = \overline{wb}^{xyz} - \overline{ \overline{w}^{xy} * \overline{b}^{xy} }^z
    wb_fact = ml_integral(w_cg*b_cg,  Hml_cg, depth, dz_cg, min_H=10.0)

    # Eddy buoyancy flux
    B_eddy = wb_avg - wb_fact

    return wb_avg, wb_fact, B_eddy, w_avg, b_avg, Hml_cg



# ============================================================
#                        DASK
# ============================================================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)


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
#                      COARSE-GRAINING FUNCTION
# ============================================================
def coarse_grain(data, window_size=12):
    """
    Coarse-grains the input data by averaging over a window of size (window_size, window_size) in the horizontal plane.
    
    Parameters:
        data: 3D xarray DataArray (time, lat, lon)
        window_size: size of the averaging window for coarse-graining (default: 4)
        
    Returns:
        Coarse-grained data
    """
    return data.coarsen(i=window_size, j=window_size, boundary="trim").mean()


# ============================================================
#                          MAIN LOOP
# ============================================================

# Build dicts indexed by date_tag
rho_dict = {os.path.basename(f).split("_")[-1].replace(".nc", ""): f
            for f in rho_files}

hml_dict = {os.path.basename(f).split("_")[-1].replace(".nc", ""): f
            for f in hml_files}

# Loop over dates that exist in both
for date_tag in sorted(rho_dict.keys()):

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

    w_matched = W_all.isel(time=t_index)

    # compute MLD quantities after coarse-graining
    wb_avg, wb_fact, B_eddy, w_avg, b_avg, Hml_cg = compute_mld_integrals_one_time(rho, Hml, w_matched)

    # save results
    ds_out = xr.Dataset({
        "wb_avg":  wb_avg,
        "wb_fact": wb_fact,
        "B_eddy":  B_eddy,
        "w_avg":   w_avg,
        "b_avg":   b_avg,
        "Hml_cg":  Hml_cg
    })

    ds_out.to_netcdf(out_file)
    print(f"Saved → {out_file}")

client.close()
cluster.close()


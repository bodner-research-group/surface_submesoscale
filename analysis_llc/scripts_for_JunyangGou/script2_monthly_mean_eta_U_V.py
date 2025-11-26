#!/usr/bin/env python3
"""
Compute monthly averages of LLC4320 surface 
- Eta (surface height anomaly), 
- U (surface zonal velocity), 
- V (surface meridional velocity)
for a Southern Ocean region spanning two faces.
"""

# ================= Imports =======================
import xarray as xr
import numpy as np
import os
import time
from dask.distributed import Client, LocalCluster

# ================= Paths =========================
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/Southern_Ocean_JunyangGou/"
os.makedirs(output_dir, exist_ok=True)

# ================= Open Dataset ===================
ds = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# ==================== Region ======================
# Face 1 region
face1 = 1
i1 = slice(1824, 4320)
j1 = slice(179, 1652)

# Face 4 region
face4 = 4
i4 = slice(0, 384)
j4 = slice(179, 1652)

# ==================== Time ========================
start_hours = 49 * 24                    # Nov 1, 2011
end_hours   = start_hours + 24*30*12     # 12 months
time_slice  = slice(start_hours, end_hours)

# ================================================================
# === FIX: coordinate definitions for staggered fields ============
coord_map = {
    "Eta": ("i", "j", "XC", "YC"),
    "U":   ("i_g", "j", "XG", "YC"),
    "V":   ("i", "j_g", "XC", "YG"),
}
# ================================================================


# ==================== Helper: extract surface field ======================
def extract_surface(ds, varname):
    print(f"Extracting {varname}...")

    i_dim, j_dim, lon_name, lat_name = coord_map[varname]

    # ------------ Face 1 ----------------
    da1 = ds[varname].isel(time=time_slice, face=face1)
    da1 = da1.isel(**{i_dim: i1, j_dim: j1})
    if "k" in da1.dims:
        da1 = da1.isel(k=0)

    # ------------ Face 4 ----------------
    da4 = ds[varname].isel(time=time_slice, face=face4)
    da4 = da4.isel(**{i_dim: i4, j_dim: j4})
    if "k" in da4.dims:
        da4 = da4.isel(k=0)

    # ------------ Stack into 1D points ----------------
    da1_stacked = da1.stack(points=(j_dim, i_dim))
    da4_stacked = da4.stack(points=(j_dim, i_dim))
    da = xr.concat([da1_stacked, da4_stacked], dim="points")

    # ------------ Lat/Lon ----------------
    lat1 = ds[lat_name].isel(face=face1, **{i_dim: i1, j_dim: j1}).stack(points=(j_dim, i_dim))
    lat4 = ds[lat_name].isel(face=face4, **{i_dim: i4, j_dim: j4}).stack(points=(j_dim, i_dim))
    lat = xr.concat([lat1, lat4], dim="points")

    lon1 = ds[lon_name].isel(face=face1, **{i_dim: i1, j_dim: j1}).stack(points=(j_dim, i_dim))
    lon4 = ds[lon_name].isel(face=face4, **{i_dim: i4, j_dim: j4}).stack(points=(j_dim, i_dim))
    lon = xr.concat([lon1, lon4], dim="points")

    # ==========================================================
    # FIX → remove MultiIndex so NetCDF can be written
    # ==========================================================
    da = da.reset_index("points")
    lat = lat.reset_index("points")
    lon = lon.reset_index("points")

    # Attach coords
    da = da.to_dataset(name=varname)
    da["lat"] = lat
    da["lon"] = lon

    return da

# ==================== Monthly averaging ======================
def compute_monthly_mean(ds_var, varname):
    """Compute monthly means robustly using string-based year-month grouping."""

    time_vals = ds["time"].isel(time=time_slice).values
    ds_var = ds_var.assign_coords(time=time_vals)

    ym = xr.DataArray(
        ds_var.time.dt.strftime("%Y-%m"),
        dims="time",
        name="month_str"
    )

    ds_monthly = ds_var.groupby(ym).mean()

    new_time = np.array([np.datetime64(m + "-01") for m in ds_monthly["month_str"].values])

    ds_monthly = ds_monthly.rename({"month_str": "time"})
    ds_monthly = ds_monthly.assign_coords(time=("time", new_time))

    return ds_monthly


# ==================== Save ======================
def save_monthly(ds_monthly, outname):
    print(f"Saving: {outname}")
    ds_monthly.to_netcdf(outname)
    print("Saved.")



# ==================== Main ======================
if __name__ == "__main__":

    cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit="11GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    varlist = [
        ("Eta", "eta_monthly.nc"),
        ("U",   "u_monthly.nc"),
        ("V",   "v_monthly.nc"),
    ]

    for varname, outfile in varlist:
        ds_var = extract_surface(ds, varname)
        ds_month = compute_monthly_mean(ds_var, varname)
        save_monthly(ds_month, os.path.join(output_dir, outfile))

    print("All variables processed.")
    client.close()
    cluster.close()

#!/usr/bin/env python3
"""
Compute monthly averages of LLC4320 surface 
- Eta (surface height anomaly), 
- U (surface zonal velocity), 
- V (surface meridional velocity)
for a Southern Ocean region spanning two faces.
Optimized: compute monthly mean per face first, then concatenate.
"""

# ================= Imports =======================
import xarray as xr
import numpy as np
import os
from dask.distributed import Client, LocalCluster

# ================= Paths =========================
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/Southern_Ocean_JunyangGou/"
os.makedirs(output_dir, exist_ok=True)

# ================= Open Dataset ===================
ds = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# ==================== Region ======================
face1 = 1
i1 = slice(1824, 4320)
j1 = slice(179, 1652)

face4 = 4
i4 = slice(0, 384)
j4 = slice(179, 1652)

# ==================== Time ========================
start_hours = 49 * 24
end_hours = start_hours + 24*30*12
time_slice = slice(start_hours, end_hours)

# ==================== Staggered coordinates =========
coord_map = {
    "Eta": ("i", "j", "XC", "YC"),
    "U":   ("i_g", "j", "XG", "YC"),
    "V":   ("i", "j_g", "XC", "YG"),
}

# ==================== Helper: extract & monthly mean per face ==========
def extract_and_monthly(ds, varname, face, i_slice, j_slice):
    """Extract surface field for a face and compute monthly mean."""
    i_dim, j_dim, lon_name, lat_name = coord_map[varname]

    da = ds[varname].isel(time=time_slice, face=face)
    da = da.isel(**{i_dim: i_slice, j_dim: j_slice})
    if "k" in da.dims:
        da = da.isel(k=0)

    # --- assign time coords ---
    da = da.assign_coords(time=ds["time"].isel(time=time_slice).values)

    # --- compute monthly mean ---
    ym = xr.DataArray(da.time.dt.strftime("%Y-%m"), dims="time", name="month_str")
    da_monthly = da.groupby(ym).mean()
    new_time = np.array([np.datetime64(m + "-01") for m in da_monthly["month_str"].values])
    da_monthly = da_monthly.rename({"month_str": "time"}).assign_coords(time=("time", new_time))

    # --- lat/lon ---
    lat = ds[lat_name].isel(face=face, **{i_dim: i_slice, j_dim: j_slice})
    lon = ds[lon_name].isel(face=face, **{i_dim: i_slice, j_dim: j_slice})

    # stack points
    da_monthly = da_monthly.stack(points=(j_dim, i_dim))
    lat = lat.stack(points=(j_dim, i_dim))
    lon = lon.stack(points=(j_dim, i_dim))

    # remove MultiIndex
    da_monthly = da_monthly.reset_index("points")
    lat = lat.reset_index("points")
    lon = lon.reset_index("points")

    # attach coords
    ds_out = da_monthly.to_dataset(name=varname)
    ds_out["lat"] = lat
    ds_out["lon"] = lon

    return ds_out

# ==================== Main ======================
if __name__ == "__main__":
    cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit="11GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    varlist = [
        ("Eta", "eta_monthly.nc"),
        ("U", "u_monthly.nc"),
        ("V", "v_monthly.nc"),
    ]

    for varname, outfile in varlist:
        # compute monthly mean per face
        ds_f1 = extract_and_monthly(ds, varname, face1, i1, j1)
        ds_f4 = extract_and_monthly(ds, varname, face4, i4, j4)

        # concatenate along points
        ds_month = xr.concat([ds_f1, ds_f4], dim="points").sortby("lon")

        # save
        print(f"Saving: {outfile}")
        ds_month.to_netcdf(os.path.join(output_dir, outfile))
        print("Saved.")

    print("All variables processed.")
    client.close()
    cluster.close()

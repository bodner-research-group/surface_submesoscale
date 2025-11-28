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
end_hours = start_hours + 365*12
time_slice = slice(start_hours, end_hours)


# ==================== Helper: extract & monthly mean per face ==========
def extract_and_monthly(ds, varname, face, i_slice, j_slice, time_slice):
    """
    Extract surface field for a single face and compute monthly means.

    Optimized for very large datasets (PB-scale):
    - Uses Dask chunking
    - Uses resample for monthly averaging
    - Avoids expensive string operations and unnecessary stacking

    Parameters
    ----------
    ds : xarray.Dataset
        The full LLC4320 dataset
    varname : str
        Variable name: "Eta", "U", or "V"
    face : int
        LLC face index
    i_slice, j_slice : slice
        Spatial slices for the face
    time_slice : slice
        Time slice to select

    Returns
    -------
    xarray.Dataset
        Dataset with monthly means for the face, with lat/lon coordinates
    """

    # Map variable to its staggered coords
    coord_map = {
        "Eta": ("i", "j", "XC", "YC"),
        "U":   ("i_g", "j", "XG", "YC"),
        "V":   ("i", "j_g", "XC", "YG"),
    }
    i_dim, j_dim, lon_name, lat_name = coord_map[varname]

    # Extract variable for the given face and slice
    da = ds[varname].isel(time=time_slice, face=face)
    da = da.isel(**{i_dim: i_slice, j_dim: j_slice})

    # If 3D, take surface level
    if "k" in da.dims:
        da = da.isel(k=0)

    # Chunk the DataArray for Dask; time chunk = 1 month (~720 hours)
    # Adjust spatial chunks to balance memory usage
    da = da.chunk({i_dim: 512, j_dim: 512, "time": 720})

    # --- Monthly mean using resample (fast, Dask-friendly) ---
    da_monthly = da.resample(time="MS").mean()

    # Extract lat/lon for the same face and slice
    lat = ds[lat_name].isel(face=face, **{i_dim: i_slice, j_dim: j_slice})
    lon = ds[lon_name].isel(face=face, **{i_dim: i_slice, j_dim: j_slice})

    # Assign lat/lon as coordinates (keeps original dims)
    da_monthly = da_monthly.assign_coords(lat=lat, lon=lon)

    # Convert to dataset
    ds_out = da_monthly.to_dataset(name=varname)

    return ds_out

# # ==================== Staggered coordinates =========
# coord_map = {
#     "Eta": ("i", "j", "XC", "YC"),
#     "U":   ("i_g", "j", "XG", "YC"),
#     "V":   ("i", "j_g", "XC", "YG"),
# }

# def extract_and_monthly(ds, varname, face, i_slice, j_slice):
#     """Extract surface field for a face and compute monthly mean."""
#     i_dim, j_dim, lon_name, lat_name = coord_map[varname]

#     da = ds[varname].isel(time=time_slice, face=face)
#     da = da.isel(**{i_dim: i_slice, j_dim: j_slice})
#     if "k" in da.dims:
#         da = da.isel(k=0)

#     # --- assign time coords ---
#     da = da.assign_coords(time=ds["time"].isel(time=time_slice).values)

#     # --- compute monthly mean ---
#     ym = xr.DataArray(da.time.dt.strftime("%Y-%m"), dims="time", name="month_str")
#     da_monthly = da.groupby(ym).mean()
#     new_time = np.array([np.datetime64(m + "-01") for m in da_monthly["month_str"].values])
#     da_monthly = da_monthly.rename({"month_str": "time"}).assign_coords(time=("time", new_time))

#     # --- lat/lon ---
#     lat = ds[lat_name].isel(face=face, **{i_dim: i_slice, j_dim: j_slice})
#     lon = ds[lon_name].isel(face=face, **{i_dim: i_slice, j_dim: j_slice})

#     # stack points
#     da_monthly = da_monthly.stack(points=(j_dim, i_dim))
#     lat = lat.stack(points=(j_dim, i_dim))
#     lon = lon.stack(points=(j_dim, i_dim))

#     # remove MultiIndex
#     da_monthly = da_monthly.reset_index("points")
#     lat = lat.reset_index("points")
#     lon = lon.reset_index("points")

#     # attach coords
#     ds_out = da_monthly.to_dataset(name=varname)
#     ds_out["lat"] = lat
#     ds_out["lon"] = lon

#     return ds_out


# def flatten_points(ds):
#     # stack i/j into points
#     i_dim, j_dim = [d for d in ds[varname].dims if d in ["i","i_g"]][0], [d for d in ds[varname].dims if d in ["j","j_g"]][0]
#     ds_flat = ds.stack(points=(j_dim, i_dim)).reset_index("points")
#     return ds_flat


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
        # ds_f1 = extract_and_monthly(ds, varname, face1, i1, j1)
        # ds_f4 = extract_and_monthly(ds, varname, face4, i4, j4)
        ds_f1 = extract_and_monthly(ds, varname, face1, i1, j1, time_slice)
        ds_f4 = extract_and_monthly(ds, varname, face4, i4, j4, time_slice)

        # concatenate along points
        # ds_f1_flat = flatten_points(ds_f1)
        # ds_f4_flat = flatten_points(ds_f4)
        # ds_month = xr.concat([ds_f1_flat, ds_f4_flat], dim="points").sortby("lon")
        # ds_month = xr.concat([ds_f1, ds_f4], dim="points").sortby("lon")

        # Concatenate along the i-axis (longitude direction)
        da_month = xr.concat([ds_f1[varname], ds_f4[varname]], dim="i")
        lat_combined = xr.concat([ds_f1["lat"], ds_f4["lat"]], dim="i")
        lon_combined = xr.concat([ds_f1["lon"], ds_f4["lon"]], dim="i")

        # Create dataset
        ds_month = xr.Dataset(
            {
                varname: da_month,
                "lat": lat_combined,
                "lon": lon_combined
            }
        )

        # Optional: sort along longitude
        # ds_month = ds_month.sortby("lon")


        # save
        print(f"Saving: {outfile}")
        ds_month.to_netcdf(os.path.join(output_dir, outfile))
        print("Saved.")

    print("All variables processed.")
    client.close()
    cluster.close()

#!/usr/bin/env python3
"""
Compute monthly means of LLC4320 surface fields:
- Eta (surface height anomaly)
- U   (surface zonal velocity, i_g, j)
- V   (surface meridional velocity, i, j_g)

For a Southern Ocean region across two LLC faces.

- Correct staggered grid coordinates
"""

# ================= Imports =======================
import xarray as xr
import numpy as np
import os
from dask.distributed import Client, LocalCluster

# ================= Paths =========================
output_dir = (
    "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/"
    "Southern_Ocean_JunyangGou/"
)
os.makedirs(output_dir, exist_ok=True)

# ================= Open Dataset ===================
ds = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# ==================== Region ======================
# Face 1
face1 = 1
i1 = slice(1824, 4320 ,1)
j1 = slice(179, 1652, 1)

# Face 4
face4 = 4
i4 = slice(0, 384, 1)
j4 = slice(179, 1652 ,1)

# ==================== Time ========================
start_hours = 49 * 24
end_hours = start_hours + 365 * 24
time_slice = slice(start_hours, end_hours)

# ===========================================================
# ETA extraction
# ===========================================================
def extract_eta(ds, face, i_slice, j_slice, time_slice):

    da = ds["Eta"].isel(face=face, time=time_slice, i=i_slice, j=j_slice)

    if "k" in da.dims:
        da = da.isel(k=0)

    da = da.chunk({"time": 720, "i": 512, "j": 512})
    da_m = da.resample(time="MS").mean()

    lon = ds["XC"].isel(face=face, i=i_slice, j=j_slice)
    lat = ds["YC"].isel(face=face, i=i_slice, j=j_slice)

    return xr.Dataset({"Eta": da_m, "lon": lon, "lat": lat})


# ===========================================================
# U extraction  (i_g, j)
# ===========================================================
def extract_u(ds, face, i_slice, j_slice, time_slice):

    da = ds["U"].isel(face=face, time=time_slice, i_g=i_slice, j=j_slice)
    if "k" in da.dims:
        da = da.isel(k=0)

    da = da.chunk({"time": 720, "i_g": 512, "j": 512})
    da_m = da.resample(time="MS").mean()

    lon = ds["XG"].isel(face=face, i_g=i_slice, j_g=j_slice)
    lat = ds["YC"].isel(face=face, i=i_slice, j=j_slice)

    return xr.Dataset({"U": da_m, "lon": lon, "lat": lat})


# ===========================================================
# V extraction  (i, j_g)
# ===========================================================
def extract_v(ds, face, i_slice, j_slice, time_slice):

    da = ds["V"].isel(face=face, time=time_slice, i=i_slice, j_g=j_slice)
    if "k" in da.dims:
        da = da.isel(k=0)

    da = da.chunk({"time": 720, "i": 512, "j_g": 512})
    da_m = da.resample(time="MS").mean()

    lon = ds["XC"].isel(face=face, i=i_slice, j=j_slice)
    lat = ds["YG"].isel(face=face, i_g=i_slice, j_g=j_slice)

    return xr.Dataset({"V": da_m, "lon": lon, "lat": lat})


# ===========================================================
# Main execution
# ===========================================================
# if __name__ == "__main__":

cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit="11GB")
client = Client(cluster)
print("\nDask dashboard:", client.dashboard_link)

# ------------------------------------
# ETA
# ------------------------------------
print("\n=== Processing ETA ===")
f1 = extract_eta(ds, face1, i1, j1, time_slice)
f4 = extract_eta(ds, face4, i4, j4, time_slice)

eta = xr.concat([f1["Eta"], f4["Eta"]], dim="i")
lat = xr.concat([f1["lat"], f4["lat"]], dim="i")
lon = xr.concat([f1["lon"], f4["lon"]], dim="i")

ds_eta = xr.Dataset({"Eta": eta, "lat": lat, "lon": lon})
ds_eta.to_netcdf(os.path.join(output_dir, "eta_monthly.nc"))
print("Saved eta_monthly.nc")

# ------------------------------------
# U
# ------------------------------------
print("\n=== Processing U ===")
f1 = extract_u(ds, face1, i1, j1, time_slice)
f4 = extract_u(ds, face4, i4, j4, time_slice)

u = xr.concat([f1["U"], f4["U"]], dim="i_g")
lat = xr.concat([f1["lat"], f4["lat"]], dim="i")
lon = xr.concat([f1["lon"], f4["lon"]], dim="i_g")

u  = u.drop_vars("face")
lat = lat.drop_vars("face")
lon = lon.drop_vars("face")

ds_u = xr.Dataset({"U": u, "lat": lat, "lon": lon})
ds_u.to_netcdf(os.path.join(output_dir, "u_monthly.nc"))
print("Saved u_monthly.nc")

# ------------------------------------
# V
# ------------------------------------
print("\n=== Processing V ===")
f1 = extract_v(ds, face1, i1, j1, time_slice)
f4 = extract_v(ds, face4, i4, j4, time_slice)

v = xr.concat([f1["V"], f4["V"]], dim="i")
lat = xr.concat([f1["lat"], f4["lat"]], dim="i_g")
lon = xr.concat([f1["lon"], f4["lon"]], dim="i")

v  = v.drop_vars("face")
lat = lat.drop_vars("face")
lon = lon.drop_vars("face")

ds_v = xr.Dataset({"V": v, "lat": lat, "lon": lon})
ds_v.to_netcdf(os.path.join(output_dir, "v_monthly.nc"))
print("Saved v_monthly.nc")

# ------------------------------------
print("\nAll variables processed.\n")

client.close()
cluster.close()

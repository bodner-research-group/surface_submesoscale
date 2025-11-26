#!/usr/bin/env python3
# ============================================================
# Compare KPPhbl with mixed layer depth (Hml)
# Fully parallelized version
# ============================================================

import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

plt.rcParams.update({'font.size': 16})

# ============================================================
# 0. DASK CLUSTER
# ============================================================
cluster = LocalCluster(
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB",
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ============================================================
# 1. Domain parameters
# ============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ============================================================
# 2. Load grid (chunked)
# ============================================================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320",consolidated=False)
lon = ds1.XC.isel(face=face, i=i, j=j)
lat = ds1.YC.isel(face=face, i=i, j=j)

# ============================================================
# 3. Paths
# ============================================================
inputdir_KPPhbl = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg/"
inputdir_Hml = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg/"

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/KPPhbl_vs_MLD"
outdir_avg = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/KPPhbl_Hml_daily_avg"

os.makedirs(figdir, exist_ok=True)
os.makedirs(outdir_avg, exist_ok=True)

# ============================================================
# 4. Files
# ============================================================
KPPhbl_files = sorted(glob(os.path.join(inputdir_KPPhbl, "KPPhbl_24h_*.nc")))
Hml_files = sorted(glob(os.path.join(inputdir_Hml, "rho_Hml_TS_daily_*.nc")))

# Dict YYYYMMDD → filename
Hml_dict = {os.path.basename(f).split("_")[-1].replace(".nc", ""): f for f in Hml_files}

# ============================================================
# 5. Parallel per-day processing
# ============================================================
def process_single_day(fK, n, date_nodash, lon, lat):
    """Parallel worker: compute daily mean & return lazy plot data."""

    # Load KPP (chunked)
    dsK = xr.open_dataset(fK, chunks={"i": 200, "j": 200, "time": 1})
    da_KPP = dsK["KPPhbl"].isel(time=n)

    # Load Hml (chunked)
    da_Hml = -xr.open_dataset(Hml_dict[date_nodash],
                              chunks={"i": 200, "j": 200})["Hml_daily"]

    # remove boundaries
    da_Hml_crop = da_Hml.isel(i=slice(1, -1), j=slice(1, -1))
    da_KPP_crop = da_KPP.isel(i=slice(1, -1), j=slice(1, -1))

    Hml_mean = da_Hml_crop.mean()
    KPP_mean = da_KPP_crop.mean()

    return {
        "date": date_nodash,
        "Hml_mean": Hml_mean,
        "KPP_mean": KPP_mean,
        "Hml_field": da_Hml,
        "KPP_field": da_KPP,
        "lon": lon,
        "lat": lat
    }


tasks = []

for fK in KPPhbl_files:

    dsK = xr.open_dataset(fK, chunks={"i": 200, "j": 200, "time": 1})
    for n, t in enumerate(dsK.time.values):
        date_str = str(np.datetime_as_string(t, unit="D"))
        date_nodash = date_str.replace("-", "")

        if date_nodash not in Hml_dict:
            print(f"Missing Hml for {date_str}, skipping.")
            continue

        tasks.append(
            delayed(process_single_day)(
                fK, n, date_nodash, lon, lat
            )
        )

print(f"Total days to process: {len(tasks)}")

# ============================================================
# 6. Compute all in parallel
# ============================================================
results = dask.compute(*tasks)

# ============================================================
# 7. Save time series
# ============================================================
dates = []
Hml_means = []
KPP_means = []

for r in results:
    dsH = r["Hml_mean"].compute()
    dsK = r["KPP_mean"].compute()

    dates.append(np.datetime64(
        f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}"
    ))
    Hml_means.append(float(dsH))
    KPP_means.append(float(dsK))

ds_timeseries = xr.Dataset(
    {
        "Hml_mean": ("time", np.array(Hml_means)),
        "KPPhbl_mean": ("time", np.array(KPP_means)),
    },
    coords={"time": np.array(dates)},
)

out_ts = os.path.join(outdir_avg, "KPPhbl_Hml_daily_timeseries.nc")
ds_timeseries.to_netcdf(out_ts)
print(f"Saved timeseries → {out_ts}")

# ============================================================
# 8. Produce maps (serial, after compute)
# ============================================================
# vmin, vmax = 100, 600

# for r in results:
#     date = r["date"]
#     date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:]}"

#     H = r["Hml_field"].compute()
#     K = r["KPP_field"].compute()

#     fig, ax = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

#     p1 = ax[0].pcolormesh(r["lon"], r["lat"], H, cmap="gist_ncar", vmin=vmin, vmax=vmax)
#     ax[0].set_title(f"Hml – {date_fmt}")
#     fig.colorbar(p1, ax=ax[0])

#     p2 = ax[1].pcolormesh(r["lon"], r["lat"], K, cmap="gist_ncar", vmin=vmin, vmax=vmax)
#     ax[1].set_title(f"KPPhbl – {date_fmt}")
#     fig.colorbar(p2, ax=ax[1])

#     plt.savefig(os.path.join(figdir, f"KPPhbl_vs_Hml_{date}.png"), dpi=120)
#     plt.close()

# ============================================================
# 9. Plot timeseries
# ============================================================
plt.figure(figsize=(14, 6))
plt.plot(ds_timeseries.time, ds_timeseries.Hml_mean, label="Hml", lw=2)
plt.plot(ds_timeseries.time, ds_timeseries.KPPhbl_mean, label="KPPhbl", lw=2)
plt.legend()
plt.grid()
plt.title("Daily Mean Depths")
plt.savefig(os.path.join(figdir, "KPPhbl_Hml_daily_timeseries.png"), dpi=150)
plt.close()

client.close()
cluster.close()
print("All done. Cluster closed cleanly.")

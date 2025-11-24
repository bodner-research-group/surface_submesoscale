###### Compare KPPhbl with mixed layer depth (Hml)
###### KPPhbl |KPP boundary layer depth, bulk Ri criterion
import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster
import dask

# ============================================================
# 0. DASK CLUSTER
# ============================================================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
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
# 2. Load grid
# ============================================================
ds_grid = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lon = ds_grid.XC.isel(face=face, i=i, j=j)
lat = ds_grid.YC.isel(face=face, i=i, j=j)

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

# Dictionary: YYYYMMDD → Hml_ncfile
Hml_dict = {os.path.basename(f).split("_")[-1].replace(".nc", ""): f for f in Hml_files}

# ============================================================
# 5. Determine common colorbar limits
# ============================================================
def get_minmax(files, varname):
    mins, maxs = [], []
    for f in files:
        ds = xr.open_dataset(f, chunks={})
        v = ds[varname]
        mins.append(v.min().compute().item())
        maxs.append(v.max().compute().item())
    return min(mins), max(maxs)

print("Computing colorbar limits...")

Hml_min, Hml_max = get_minmax(Hml_files, "Hml_daily")
KPP_min, KPP_max = get_minmax(KPPhbl_files, "KPPhbl")

# use shared colorbar range
vmin = min(Hml_min, KPP_min)
vmax = max(Hml_max, KPP_max)

print("Colorbar limits:", vmin, vmax)

# ========= BEFORE MAIN LOOP =========
dates = []
Hml_means = []
KPP_means = []

# ============================================================
# 6. Main loop
# ============================================================
for fK in KPPhbl_files:

    dsK = xr.open_dataset(fK, chunks={})
    time_vals = dsK.time.values  # 7 days
    da_KPP = dsK["KPPhbl"].isel(face=face, i=i, j=j)

    for n, t in enumerate(time_vals):

        # date formats
        date_str = str(np.datetime_as_string(t, unit="D"))
        date_nodash = date_str.replace("-", "")

        # check for Hml file
        if date_nodash not in Hml_dict:
            print(f"Missing Hml for {date_str}, skipping.")
            continue

        # read Hml
        fH = Hml_dict[date_nodash]
        da_Hml = -xr.open_dataset(fH, chunks={})["Hml_daily"].isel(i=i, j=j)

        # select 1 day of KPPhbl
        kpp_day = da_KPP.isel(time=n)

        # =====================================================
        # 6A. EXCLUDE ONE GRID-CELL BOUNDARY
        # =====================================================
        da_Hml_crop = da_Hml.isel(i=slice(1, -1), j=slice(1, -1))
        kpp_day_crop = kpp_day.isel(i=slice(1, -1), j=slice(1, -1))

        # =====================================================
        # 6B. DAILY AVERAGES
        # =====================================================
        Hml_mean = float(da_Hml_crop.mean().compute())
        KPP_mean = float(kpp_day_crop.mean().compute())

        # store values
        dates.append(np.datetime64(date_str))
        Hml_means.append(Hml_mean)
        KPP_means.append(KPP_mean)

        # =====================================================
        # 6C. FIGURE
        # =====================================================
        fig, ax = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

        # Hml
        p1 = ax[0].pcolormesh(lon, lat, da_Hml, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[0].set_title(f"Hml – {date_str}\nDaily mean = {Hml_mean:.2f} m")
        cb1 = fig.colorbar(p1, ax=ax[0], label="Depth (m)")

        # KPPhbl
        p2 = ax[1].pcolormesh(lon, lat, kpp_day, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[1].set_title(f"KPPhbl – {date_str}\nDaily mean = {KPP_mean:.2f} m")
        cb2 = fig.colorbar(p2, ax=ax[1], label="Depth (m)")

        # save
        outname = os.path.join(figdir, f"KPPhbl_vs_Hml_{date_nodash}.png")
        plt.savefig(outname, dpi=150)
        plt.close(fig)

        print(f"Saved figure: {outname}")


# ============================================================
# AFTER ALL LOOPS: SAVE ONE NETCDF FILE
# ============================================================
ds_timeseries = xr.Dataset(
    {
        "Hml_mean": ("time", np.array(Hml_means)),
        "KPPhbl_mean": ("time", np.array(KPP_means)),
    },
    coords={"time": ("time", np.array(dates))},
)

out_ts = os.path.join(outdir_avg, "KPPhbl_Hml_daily_timeseries.nc")
ds_timeseries.to_netcdf(out_ts)

print(f"Saved full daily mean timeseries → {out_ts}")

client.close()
cluster.close()

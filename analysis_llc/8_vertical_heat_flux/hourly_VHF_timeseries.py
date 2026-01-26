import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import dask
import dask.array as da
from dask import delayed
from dask.distributed import Client

# ============================================================
# USER PARAMETERS
# ============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

shortname = "hourly_wT_gaussian_30km"
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/{shortname}"
boundary = 2

ts_outfile = os.path.join(data_dir, f"{shortname}_timeseries.nc")

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/{shortname}"
os.makedirs(figdir, exist_ok=True)

plt.rcParams.update({"font.size": 16})

# ============================================================
# START DASK CLIENT
# ============================================================
client = Client(
    processes=True,
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB",
)
print(client)

# ============================================================
# LOAD FILE LIST
# ============================================================
nc_files = sorted(glob(os.path.join(data_dir, "wT_eddy_*.nc")))
if len(nc_files) == 0:
    raise FileNotFoundError("No wT_eddy_*.nc files found!")

print(f"Found {len(nc_files)} hourly files")

# ============================================================
# DELAYED WORKER FUNCTION (SAFE)
# ============================================================
@delayed
def process_file(path):
    date_tag = os.path.basename(path).split("_")[-1].replace(".nc", "")
    date_part, hour_part = date_tag.rsplit("-", 1)
    dt = np.datetime64(f"{date_part}T{hour_part}:00", "h")

    ds = xr.open_dataset(path, chunks={"i": -1, "j": -1})

    slicer = dict(
        j=slice(boundary, -boundary),
        i=slice(boundary, -boundary),
    )

    wT_total = ds["wT_total"].isel(**slicer).mean(skipna=True)
    wT_mean  = ds["wT_mean"].isel(**slicer).mean(skipna=True)
    wT_eddy  = ds["wT_eddy"].isel(**slicer).mean(skipna=True)

    return dt, wT_total, wT_mean, wT_eddy

# ============================================================
# BUILD & EXECUTE GRAPH
# ============================================================
tasks = [process_file(path) for path in nc_files]

print("Computing...")
results = dask.compute(*tasks)

# ============================================================
# UNPACK + SORT
# ============================================================
results = sorted(results, key=lambda x: x[0])

time, wT_total, wT_mean, wT_eddy = zip(*results)

# ============================================================
# FINAL COMPUTE + CONVERT TO FLOATS
# ============================================================
wT_total = [float(x.values) for x in da.compute(*wT_total)]
wT_mean  = [float(x.values) for x in da.compute(*wT_mean)]
wT_eddy  = [float(x.values) for x in da.compute(*wT_eddy)]

# ============================================================
# SAVE TIMESERIES DATASET
# ============================================================
ts = xr.Dataset(
    {
        "wT_total_mean": ("time", wT_total),
        "wT_mean_mean":  ("time", wT_mean),
        "wT_eddy_mean":  ("time", wT_eddy),
    },
    coords={"time": np.array(time)},
)

ts.to_netcdf(ts_outfile)
print(f"Saved timeseries → {ts_outfile}")

# ============================================================
# FIGURE (ALL TERMS)
# ============================================================
figfile = os.path.join(figdir, f"{shortname}_all_timeseries.png")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts.time, ts.wT_total_mean, label=r"$\langle wT \rangle$")
ax.plot(ts.time, ts.wT_mean_mean, label=r"$\langle \bar{w}\bar{b} \rangle$")
ax.plot(ts.time, ts.wT_eddy_mean, label=r"$\langle w'b' \rangle$")

ax.axhline(0, color="k", lw=0.7)
ax.set_title("Horizontal Mean (⟨⋅⟩ₓᵧ)")
ax.set_xlabel("Time")
ax.grid(alpha=0.3)
ax.legend(ncol=2)

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

# ============================================================
# FIGURE (EDDY ONLY)
# ============================================================
figfile = os.path.join(figdir, f"{shortname}_eddy_timeseries.png")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts.time, ts.wT_eddy_mean, label=r"$\langle w'b' \rangle$")
ax.axhline(0, color="k", lw=0.7)

ax.set_title("Horizontal Mean (⟨⋅⟩ₓᵧ)")
ax.set_xlabel("Time")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

# ============================================================
# CLEANUP
# ============================================================
client.close()

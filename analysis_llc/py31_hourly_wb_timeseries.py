import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# ============================================================
# USER PARAMETERS
# ============================================================
# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain



data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/hourly_wb_eddy_1_12deg"
boundary = 2

ts_outfile = os.path.join(data_dir, "hourly_wb_eddy_1_12deg_timeseries.nc")

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/hourly_wb_eddy_1_12deg"
os.makedirs(figdir, exist_ok=True)
figfile = os.path.join(figdir, "hourly_wb_eddy_1_12deg_timeseries.png")

plt.rcParams.update({'font.size': 16})

# ============================================================
# LOAD ALL DAILY FILES
# ============================================================
nc_files = sorted(glob(os.path.join(data_dir, "wb_eddy_*.nc")))
if len(nc_files) == 0:
    raise FileNotFoundError("No wb_eddy_*.nc files found!")
print(f"Found {len(nc_files)} hourly files.")

# ============================================================
# PREPARE STORAGE
# ============================================================
datetime = []

wb_total_list = []
wb_mean_list  = []
wb_eddy_list  = []

# ============================================================
# LOOP OVER FILES
# ============================================================
for path in nc_files:

    date_tag = os.path.basename(path).split("_")[-1].replace(".nc", "")
    
    # date_tag example: '2012-01-01-00'
    date_part, hour_part = date_tag.rsplit("-", 1)

    dt = np.datetime64(f"{date_part}T{hour_part}:00", "h")
    datetime.append(dt)

    ds = xr.open_dataset(path)

    wb_total = ds["wb_avg"]
    wb_mean  = ds["wb_fact"]
    wb_eddy  = ds["B_eddy"]

    # --------------------------------------------------------
    # Remove boundary
    # --------------------------------------------------------
    slicer = dict(
        j=slice(boundary, -boundary),
        i=slice(boundary, -boundary)
    )

    wb_total_inner = wb_total.isel(**slicer)
    wb_mean_inner  = wb_mean.isel(**slicer)
    wb_eddy_inner  = wb_eddy.isel(**slicer)

    # --------------------------------------------------------
    # Horizontal means
    # --------------------------------------------------------
    wb_total_list.append(float(wb_total_inner.mean(skipna=True)))
    wb_mean_list.append(float(wb_mean_inner.mean(skipna=True)))
    wb_eddy_list.append(float(wb_eddy_inner.mean(skipna=True)))


    print(
        f"{date_tag}: "
        f"<wb>={wb_total_list[-1]:.3e} "
    )

# ============================================================
# DATASET OUTPUT
# ============================================================
ts = xr.Dataset(
    {
        "wb_total_mean":         ("time", wb_total_list),
        "wb_mean_mean":          ("time", wb_mean_list),
        "wb_eddy_mean":          ("time", wb_eddy_list)
    },
    coords={"time": np.array(datetime)}
)

ts.to_netcdf(ts_outfile)
print(f"Saved timeseries → {ts_outfile}")

# ============================================================
# FIGURE
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(ts.time, ts.wb_total_mean,
        label=r"$\langle\overline{\overline{wb}^{xy}}^z\rangle$")

ax.plot(ts.time, ts.wb_mean_mean,
        label=r"$\langle\overline{\overline{w}^{xy}}^z\overline{\overline{b}^{xy}}^z\rangle$")

ax.plot(ts.time, ts.wb_eddy_mean,
        label=r"$\langle w'b'\rangle = \langle\overline{\overline{wb}^{xy}}^z - \overline{\overline{w}^{xy}\overline{b}^{xy}}^z\rangle$")

ax.axhline(0, color='k', lw=0.7)

ax.set_title("Horizontal Mean (⟨⋅⟩ₓᵧ)")
ax.set_xlabel("Time")
ax.grid(alpha=0.3)
ax.legend(ncol=2)

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

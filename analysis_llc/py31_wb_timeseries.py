import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# ============================================================
# USER PARAMETERS
# ============================================================
from set_constant import domain_name, face, i, j

data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_daily"
boundary = 2

ts_outfile = os.path.join(data_dir, "wb_mld_horizontal_timeseries.nc")
figfile = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_mld_daily/wb_mld_horizontal_timeseries.png"

# ============================================================
# LOAD ALL DAILY FILES
# ============================================================
nc_files = sorted(glob(os.path.join(data_dir, "wb_mld_daily_*.nc")))
if len(nc_files) == 0:
    raise FileNotFoundError("No wb_mld_daily_*.nc files found!")
print(f"Found {len(nc_files)} daily files.")

# ============================================================
# PREPARE STORAGE
# ============================================================
dates = []
# regular means
wb_avg_list = []
wb_fact_list = []
wb_eddy_list = []
# absolute means
wb_avg_abs_list = []
wb_fact_abs_list = []
wb_eddy_abs_list = []

# ============================================================
# LOOP
# ============================================================
for path in nc_files:

    date_tag = os.path.basename(path).split("_")[-1].replace(".nc", "")
    dates.append(np.datetime64(date_tag))

    ds = xr.open_dataset(path)

    wb_avg  = ds["wb_avg"]
    wb_fact = ds["wb_fact"]
    wb_eddy = wb_avg - wb_fact

    # remove boundary
    slicer = dict(j=slice(boundary, -boundary), i=slice(boundary, -boundary))

    wb_avg_inner  = wb_avg.isel(**slicer)
    wb_fact_inner = wb_fact.isel(**slicer)
    wb_eddy_inner = wb_eddy.isel(**slicer)

    # regular means
    wb_avg_list.append(float(wb_avg_inner.mean()))
    wb_fact_list.append(float(wb_fact_inner.mean()))
    wb_eddy_list.append(float(wb_eddy_inner.mean()))

    # absolute means
    wb_avg_abs_list.append(float(np.abs(wb_avg_inner).mean()))
    wb_fact_abs_list.append(float(np.abs(wb_fact_inner).mean()))
    wb_eddy_abs_list.append(float(np.abs(wb_eddy_inner).mean()))

    print(f"{date_tag}: "
          f"mean wb={wb_avg_list[-1]:.3e}, abs={wb_avg_abs_list[-1]:.3e}")

# ============================================================
# DATASET OUTPUT
# ============================================================
ts = xr.Dataset(
    {
        # mean values
        "wb_avg_mean":  ("time", wb_avg_list),
        "wb_fact_mean": ("time", wb_fact_list),
        "wb_eddy_mean": ("time", wb_eddy_list),

        # absolute means
        "wb_avg_abs_mean":  ("time", wb_avg_abs_list),
        "wb_fact_abs_mean": ("time", wb_fact_abs_list),
        "wb_eddy_abs_abs_mean": ("time", wb_eddy_abs_list),
    },
    coords={"time": np.array(dates)}
)

ts.to_netcdf(ts_outfile)
print(f"Saved timeseries → {ts_outfile}")

# ============================================================
# TWO-PANEL FIGURE
# ============================================================
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- Panel 1: regular means ---
axs[0].plot(ts.time, ts.wb_avg_mean,  label="⟨w̄b̄⟩")
axs[0].plot(ts.time, ts.wb_fact_mean, label="⟨w̄⟩⟨b̄⟩")
axs[0].plot(ts.time, ts.wb_eddy_mean, label="eddy term")
axs[0].axhline(0, color='k', lw=0.7)
axs[0].set_title("Horizontal Mean (⟨⋅⟩ₓᵧ)")
axs[0].grid(alpha=0.3)
axs[0].legend()

# --- Panel 2: absolute means ---
axs[1].plot(ts.time, ts.wb_avg_abs_mean,  label="⟨|w̄b̄|⟩")
axs[1].plot(ts.time, ts.wb_fact_abs_mean, label="⟨|w̄| |b̄|⟩")
axs[1].plot(ts.time, ts.wb_eddy_abs_abs_mean, label="⟨|eddy term|⟩")
axs[1].axhline(0, color='k', lw=0.7)
axs[1].set_title("Horizontal Mean of Absolute Value (⟨|⋅|⟩ₓᵧ)")
axs[1].grid(alpha=0.3)
axs[1].legend()

axs[1].set_xlabel("Time")

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

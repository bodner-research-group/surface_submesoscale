import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# ============================================================
# USER PARAMETERS
# ============================================================
from set_constant import domain_name, face, i, j

data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_daily_1_4deg"
boundary = 2

ts_outfile = os.path.join(data_dir, "wb_mld_horizontal_timeseries_1_4deg.nc")
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_mld_daily_1_4deg"
os.makedirs(figdir, exist_ok=True)
figfile = os.path.join(figdir, "wb_mld_horizontal_timeseries_1_4deg.png")

plt.rcParams.update({'font.size': 16})

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
wb_total_list = []
wb_mean_list  = []
wb_eddy_list  = []
delta_wb_eddy_list = []   # NEW: <wb'_surf - wb'_mlb>

# ============================================================
# LOOP OVER FILES
# ============================================================
for path in nc_files:

    date_tag = os.path.basename(path).split("_")[-1].replace(".nc", "")
    dates.append(np.datetime64(f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:]}", 'D'))

    ds = xr.open_dataset(path)

    wb_total = ds["wb_avg"]
    wb_mean  = ds["wb_fact"]
    wb_eddy  = ds["B_eddy"]

    wb_eddy_surf = ds["wb_eddy_surf"]
    wb_eddy_mlb  = ds["wb_eddy_mlb"]

    # --------------------------------------------------------
    # Remove boundary
    # --------------------------------------------------------
    slicer = dict(j=slice(boundary, -boundary),
                  i=slice(boundary, -boundary))

    wb_total_inner = wb_total.isel(**slicer)
    wb_mean_inner  = wb_mean.isel(**slicer)
    wb_eddy_inner  = wb_eddy.isel(**slicer)

    wb_eddy_surf_inner = wb_eddy_surf.isel(**slicer)
    wb_eddy_mlb_inner  = wb_eddy_mlb.isel(**slicer)

    # --------------------------------------------------------
    # Horizontal means
    # --------------------------------------------------------
    wb_total_list.append(float(wb_total_inner.mean(skipna=True)))
    wb_mean_list.append(float(wb_mean_inner.mean(skipna=True)))
    wb_eddy_list.append(float(wb_eddy_inner.mean(skipna=True)))

    delta_wb_eddy = wb_eddy_surf_inner - wb_eddy_mlb_inner
    delta_wb_eddy_list.append(float(delta_wb_eddy.mean(skipna=True)))

    print(f"{date_tag}: mean wb = {wb_total_list[-1]:.3e}, "
          f"Δwb'_surf-mlb = {delta_wb_eddy_list[-1]:.3e}")

# ============================================================
# DATASET OUTPUT
# ============================================================
ts = xr.Dataset(
    {
        "wb_total_mean":        ("time", wb_total_list),
        "wb_mean_mean":         ("time", wb_mean_list),
        "wb_eddy_mean":         ("time", wb_eddy_list),
        "delta_wb_eddy_mean":   ("time", delta_wb_eddy_list),
    },
    coords={"time": np.array(dates)}
)

ts.to_netcdf(ts_outfile)
print(f"Saved timeseries → {ts_outfile}")

# ============================================================
# FIGURE
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(11, 6))

ax.plot(ts.time, ts.wb_total_mean,
        label=r"$\langle\overline{\overline{wb}^{xy}}^z\rangle$")

ax.plot(ts.time, ts.wb_mean_mean,
        label=r"$\langle\overline{\overline{w}^{xy}}^z\overline{\overline{b}^{xy}}^z\rangle$")

ax.plot(ts.time, ts.wb_eddy_mean,
        label=r"$\langle\overline{\overline{wb}^{xy}}^z - \overline{\overline{w}^{xy}\overline{b}^{xy}}^z\rangle$")

ax.plot(ts.time, - ts.delta_wb_eddy_mean,
        lw=2.5, linestyle="--",
        label=r"$\langle w'b'|_{\mathrm{mlb}} - w'b'|_{\mathrm{surf}}\rangle$")

ax.axhline(0, color='k', lw=0.7)

ax.set_title("Horizontal Mean (⟨⋅⟩ₓᵧ)")
ax.set_xlabel("Time")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

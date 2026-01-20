import xarray as xr
import matplotlib.pyplot as plt
import os

# ============================================================
# USER PARAMETERS
# ============================================================
domain_name = "icelandic_basin"
shortname = "hourly_wb_eddy_gaussian_wide"

data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/{shortname}"
figdir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/{shortname}"

ts_file = os.path.join(data_dir, f"{shortname}_timeseries.nc")
os.makedirs(figdir, exist_ok=True)

plt.rcParams.update({"font.size": 16})

# ============================================================
# LOAD TIMESERIES
# ============================================================
ds = xr.open_dataset(ts_file)

# Safety check
if "wb_eddy_mean" not in ds:
    raise KeyError("Variable 'wb_eddy_mean' not found in dataset")

# ============================================================
# DAILY AVERAGE
# ============================================================
wb_eddy_daily = (
    ds["wb_eddy_mean"]
    .resample(time="1D")
    .mean()
)

# ============================================================
# SAVE DAILY DATASET (OPTIONAL BUT RECOMMENDED)
# ============================================================
daily_outfile = os.path.join(
    data_dir, f"{shortname}_eddy_daily_timeseries.nc"
)

wb_eddy_daily.to_dataset(name="wb_eddy_daily_mean").to_netcdf(daily_outfile)
print(f"Saved daily-averaged timeseries → {daily_outfile}")

# ============================================================
# PLOT
# ============================================================
figfile = os.path.join(
    figdir, f"{shortname}_eddy_daily_timeseries.png"
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    wb_eddy_daily.time,
    wb_eddy_daily,
    lw=2,
    label=r"Daily $\langle w'b' \rangle$"
)

ax.axhline(0, color="k", lw=0.8)
ax.set_title("Daily-Averaged Eddy Buoyancy Flux\nHorizontal Mean (⟨⋅⟩ₓᵧ)")
ax.set_xlabel("Time")
ax.set_ylabel(r"$w'b'$")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

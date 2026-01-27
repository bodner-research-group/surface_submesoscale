import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# USER PARAMETERS
# ============================================================
domain_name = "icelandic_basin"
shortname = "hourly_wT_gaussian_30km"

data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/{shortname}"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/VHF_theory"

os.makedirs(figdir, exist_ok=True)

ts_file = os.path.join(data_dir, f"{shortname}_timeseries.nc")

# Physical constants
rho0 = 1027.5      # kg/m^3
Cp   = 3995.0      # J/(kg K)

plt.rcParams.update({"font.size": 16})

# ============================================================
# LOAD HOURLY TIMESERIES
# ============================================================
ds = xr.open_dataset(ts_file)

# ============================================================
# CONVERT TO HEAT FLUX (W/m^2)
# ============================================================
Q_total = rho0 * Cp * ds["wT_total_mean"]
Q_eddy  = rho0 * Cp * ds["wT_eddy_mean"]

# ============================================================
# DAILY AVERAGING
# ============================================================
Q_total_daily = Q_total.resample(time="1D").mean()
Q_eddy_daily  = Q_eddy.resample(time="1D").mean()

# ============================================================
# SAVE DAILY TIMESERIES
# ============================================================
daily_ds = xr.Dataset(
    {
        "Q_total_daily": Q_total_daily,
        "Q_eddy_daily":  Q_eddy_daily,
    }
)

daily_outfile = os.path.join(data_dir, f"{shortname}_daily_heatflux.nc")
daily_ds.to_netcdf(daily_outfile)

print(f"Saved daily heat flux → {daily_outfile}")

# ============================================================
# PLOT: TOTAL + EDDY HEAT FLUX
# ============================================================
figfile = os.path.join(figdir, f"{shortname}_daily_heatflux.png")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    daily_ds.time,
    daily_ds.Q_total_daily,
    label=r"$\rho_0 C_p \langle w\theta \rangle$",
    lw=2,
)

ax.plot(
    daily_ds.time,
    daily_ds.Q_eddy_daily,
    label=r"$\rho_0 C_p \langle w'\theta' \rangle$",
    lw=2,
)

ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Heat Flux [W m$^{-2}$]")
ax.set_xlabel("Time")
ax.set_title("Daily-Averaged Vertical Heat Flux")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

# ============================================================
# OPTIONAL: EDDY-ONLY FIGURE
# ============================================================
figfile = os.path.join(figdir, f"{shortname}_daily_eddy_heatflux.png")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    daily_ds.time,
    daily_ds.Q_eddy_daily,
    label=r"$\rho_0 C_p \langle w'\theta' \rangle$",
    lw=2,
)

ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Heat Flux [W m$^{-2}$]")
ax.set_xlabel("Time")
ax.set_title("Daily-Averaged Eddy Vertical Heat Flux")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figfile, dpi=150)
plt.close()

print(f"Saved figure → {figfile}")

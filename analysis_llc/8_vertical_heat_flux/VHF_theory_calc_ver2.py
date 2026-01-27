import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.rcParams.update({'font.size': 18})  # Global font size setting for figures

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/VHF_theory"
os.makedirs(figdir, exist_ok=True)

# ========== Physical constants ==========
rho0 = 1027.5
Cp = 3995
Ce = 0.06
mu0 = 44 / 63
g = 9.81
alpha = 1.4e-4   # winter average
f0 = 1.27e-4

Const = rho0 * Cp * Ce * mu0 * g / (2 * alpha * f0)

# ========== Horizontal Turner Angle ==========
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_Timeseries_Stats.nc"
TuH_means = xr.open_dataset(fname).TuH_means

fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory/TuH_domain_avg_timeseries.nc"
# Tu_calc = xr.open_dataset(fname).tan_mean_TuH_plus1
Tu_calc = xr.open_dataset(fname).mean_tan_TuH_plus1

# ========== Steric |∇η′|² ==========
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly_timeseries/grad2_timeseries.nc"
eta_prime_grad2_mean = xr.open_dataset(fname).eta_prime_grad2_mean

# ========== SSH submesoscale ==========
fname14 = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_Gaussian_submeso_LambdaMLI_timeseries.nc"
eta_submeso_grad2 = xr.open_dataset(fname14).eta_submeso_grad2_mean

# ========== Diagnosed Eddy Heat Flux ==========
eddy_ds = xr.open_dataset(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/"
    f"{domain_name}/hourly_wT_gaussian_30km/"
    f"hourly_wT_gaussian_30km_daily_heatflux.nc"
)

Q_eddy_daily = eddy_ds["Q_eddy_daily"]

# ========== Align all time series ==========
Tu_calc_aligned, eta_submeso_grad2_aligned, eta_prime_grad2_mean_aligned, Q_eddy_daily_aligned = xr.align(
    Tu_calc,
    eta_submeso_grad2,
    eta_prime_grad2_mean,
    Q_eddy_daily,
    join="inner"
)

# ========== Theory-predicted VHF ==========
VHF_theory_submeso = Const * eta_submeso_grad2_aligned * Tu_calc_aligned
VHF_theory_steric  = Const * eta_prime_grad2_mean_aligned * Tu_calc_aligned

# =======================
# Plot time series
# =======================
dates = pd.to_datetime(VHF_theory_submeso.time.values)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    dates,
    VHF_theory_submeso.values,
    label="Theory VHF (Submesoscale SSH)",
    color="tab:blue",
    linewidth=2
)

ax.plot(
    dates,
    VHF_theory_steric.values,
    label="Theory VHF (Steric Height)",
    color="tab:red",
    linestyle="--",
    linewidth=2
)

ax.plot(
    dates,
    Q_eddy_daily_aligned.values,
    label="Diagnosed Eddy Heat Flux",
    color="k",
    linewidth=2.5
)

ax.axhline(0, color="k", lw=0.8)

ax.set_title("Vertical Heat Flux: Theory vs Diagnosed Eddy Flux")
ax.set_ylabel("VHF (W m$^{-2}$)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

plt.tight_layout()

# =======================
# Save figure
# =======================
fig_path = os.path.join(figdir, "VHF_theory.png")
fig.savefig(fig_path, dpi=300)

print(f"\nSaved figure to: {fig_path}")

# =======================
# Jan–Mar averages
# =======================
eta_submeso_grad2_aligned['time'] = pd.to_datetime(eta_submeso_grad2_aligned.time.values)
Tu_calc_aligned['time'] = pd.to_datetime(Tu_calc_aligned.time.values)

jan_mar = eta_submeso_grad2_aligned.time.dt.month.isin([1, 2, 3])

eta_submeso_JFM_mean = eta_submeso_grad2_aligned.sel(time=jan_mar).mean(dim="time")
Tu_calc_JFM_mean     = Tu_calc_aligned.sel(time=jan_mar).mean(dim="time")
Q_eddy_JFM_mean      = Q_eddy_daily_aligned.sel(time=jan_mar).mean(dim="time")

# =======================
# Print results
# =======================
print("========== Constants ==========")
print(f"Const = {Const:.4e}")

print("\n========== Jan–Mar Averages ==========")
print(f"<eta_submeso_grad2>_JFM = {eta_submeso_JFM_mean.values:.4e}")
print(f"<Tu_calc>_JFM           = {Tu_calc_JFM_mean.values:.4e}")
print(f"<Q_eddy>_JFM            = {Q_eddy_JFM_mean.values:.4e} W m^-2")

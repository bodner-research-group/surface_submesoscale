import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.rcParams.update({'font.size': 18})

# =====================
# Domain
# =====================
domain_name = "icelandic_basin"
face = 2

# larger domain
i = slice(527, 1007)
j = slice(2960, 3441)

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}/VHF_theory"
os.makedirs(figdir, exist_ok=True)

# =====================
# Physical constants
# =====================
rho0 = 1027.5
Cp = 3995
Ce = 0.06
mu0 = 44 / 63
g = 9.81
alpha = 1.6011e-04   # annual mean
# alpha = 1.4e-4     # winter average
f0 = 1.27e-4

Const = rho0 * Cp * Ce * mu0 * g / (alpha * f0)

# =====================
# Load theory input data
# =====================
buoyancy_ratio_means = xr.open_dataset(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory/TurnerAngle_BuoyancyRatio_daily_Timeseries.nc"
).buoyancy_ratio_means

eta_steric_grad_mag = xr.open_dataset(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory/eta_steric_grad_mag_daily.nc"
).eta_prime_grad_mag.sel(i=i, j=j)

eta_submeso_grad_mag = xr.open_dataset(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory/eta_submeso_grad_mag_daily.nc"
).eta_submeso_grad_mag.sel(i=i, j=j)

# =====================
# Load DAILY EDDY HEAT FLUX
# =====================
eddy_ds = xr.open_dataset(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/"
    f"{domain_name}/hourly_wT_gaussian_30km/"
    f"hourly_wT_gaussian_30km_daily_heatflux.nc"
)

Q_eddy_daily = eddy_ds["Q_eddy_daily"]

# =====================
# Align in time
# =====================
buoyancy_ratio_means, eta_steric_grad_mag, eta_submeso_grad_mag, Q_eddy_daily = xr.align(
    buoyancy_ratio_means,
    eta_steric_grad_mag,
    eta_submeso_grad_mag,
    Q_eddy_daily,
    join="inner"
)

# =====================
# Theory VHF (maps)
# =====================

VHF_theory_submeso_map = Const * (eta_submeso_grad_mag ** 2) 
VHF_theory_steric_map  = Const * (eta_steric_grad_mag ** 2) 

# =====================
# Domain average
#   exclude 2 grid points at boundaries
# =====================
VHF_theory_submeso = (
    VHF_theory_submeso_map
    .isel(i=slice(2, -2), j=slice(2, -2))
    .mean(dim=("i", "j"))
)

VHF_theory_steric = (
    VHF_theory_steric_map
    .isel(i=slice(2, -2), j=slice(2, -2))
    .mean(dim=("i", "j"))
)

from scipy.io import savemat

buoyancy_ratio_means = buoyancy_ratio_means.rename({'date': 'time'})

VHF_theory_submeso = VHF_theory_submeso * buoyancy_ratio_means
VHF_theory_steric_map = VHF_theory_steric_map * buoyancy_ratio_means



# =====================
# Save to NetCDF instead of .mat
# =====================

dates = pd.to_datetime(VHF_theory_submeso.time.values)

ds_out = xr.Dataset(
    data_vars=dict(
        VHF_theory_submeso=("time", VHF_theory_submeso.values),
        VHF_theory_steric=("time", VHF_theory_steric.values),
        Q_eddy_daily=("time", Q_eddy_daily.values),
    ),
    coords=dict(
        time=VHF_theory_submeso.time
    ),
    attrs=dict(
        description="Domain-averaged VHF theory and diagnosed eddy heat flux",
        domain=domain_name,
        alpha=alpha,
        f0=f0,
        Const=Const,
        units="W m-2"
    )
)

data_output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/Manuscript_Data/{domain_name}/"

nc_path = os.path.join(data_output_dir, "VHF_timeseries_MeanAlpha_DailyBuoyancyRatio.nc")
ds_out.to_netcdf(nc_path)

print(f"Saved NetCDF file to: {nc_path}")


# =====================
# Plot time series
# =====================

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
    Q_eddy_daily.values,
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

# =====================
# Save
# =====================
fig_path = os.path.join(figdir, "VHF_theory_domain_MeanAlpha_DailyBuoyancyRatio.png")
fig.savefig(fig_path, dpi=300)

print(f"\nSaved figure to: {fig_path}")

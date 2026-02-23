import os
import glob
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
alpha_mean = 1.6011e-04
# alpha_mean = 1.406e-04
f0 = 1.27e-4

Const = rho0 * Cp * Ce * mu0 * g / (alpha_mean * f0)

# =====================
# Load SSH gradient maps
# =====================
eta_steric_grad_mag = xr.open_dataset(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory/eta_steric_grad_mag_daily.nc"
).eta_prime_grad_mag.sel(i=i, j=j)

eta_submeso_grad_mag = xr.open_dataset(
    f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory/eta_submeso_grad_mag_daily.nc"
).eta_submeso_grad_mag.sel(i=i, j=j)


# eta_submeso_grad_mag = xr.open_dataset(
#     f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory/eta_submeso_grad_mag_daily_17km.nc"
# ).eta_submeso_grad_mag.sel(i=i, j=j)



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
# Load DAILY buoyancy ratio maps (SAFE VERSION)
# =====================

import glob

br_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_BuoyancyRatio_daily"
br_files = sorted(glob.glob(os.path.join(br_dir, "TurnerAngle_BuoyancyRatio_daily_*.nc")))

datasets = []

for f in br_files:
    ds = xr.open_dataset(f)  # no parallel, no chunks
    ds = ds[["buoyancy_ratio", "buoyancy_ratio_linearEOS"]]
    datasets.append(ds)

ds_br = xr.concat(datasets, dim="time")

buoyancy_ratio = ds_br["buoyancy_ratio"].sel(i=i, j=j)
buoyancy_ratio_linear = ds_br["buoyancy_ratio_linearEOS"].sel(i=i, j=j)

buoyancy_ratio["time"] = buoyancy_ratio.indexes["time"].normalize()
buoyancy_ratio_linear["time"] = buoyancy_ratio_linear.indexes["time"].normalize()
buoyancy_ratio = buoyancy_ratio.reset_coords(drop=True)
buoyancy_ratio_linear = buoyancy_ratio_linear.reset_coords(drop=True)

# =====================
# Align in time
# =====================
(
    eta_steric_grad_mag,
    eta_submeso_grad_mag,
    buoyancy_ratio,
    buoyancy_ratio_linear,
    Q_eddy_daily,
) = xr.align(
    eta_steric_grad_mag,
    eta_submeso_grad_mag,
    buoyancy_ratio,
    buoyancy_ratio_linear,
    Q_eddy_daily,
    join="inner"
)

# =====================
# Theory VHF maps (no buoyancy ratio yet)
# =====================
VHF_theory_submeso_map = Const * (eta_submeso_grad_mag ** 2)
VHF_theory_steric_map  = Const * (eta_steric_grad_mag ** 2)

# =====================
# Multiply by buoyancy ratio MAPS
# =====================
VHF_submeso_fullEOS_map = VHF_theory_submeso_map * buoyancy_ratio
VHF_steric_fullEOS_map  = VHF_theory_steric_map  * buoyancy_ratio

VHF_submeso_linear_map = VHF_theory_submeso_map * buoyancy_ratio_linear
VHF_steric_linear_map  = VHF_theory_steric_map  * buoyancy_ratio_linear

# =====================
# Domain mean (exclude 2 boundary points)
# =====================
def domain_mean(da):
    return (
        da.isel(i=slice(2, -2), j=slice(2, -2))
          .mean(dim=("i", "j"))
    )

VHF_submeso_fullEOS = domain_mean(VHF_submeso_fullEOS_map)
VHF_steric_fullEOS  = domain_mean(VHF_steric_fullEOS_map)

VHF_submeso_linear = domain_mean(VHF_submeso_linear_map)
VHF_steric_linear  = domain_mean(VHF_steric_linear_map)

# =====================
# Save NetCDF
# =====================
data_output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/Manuscript_Data/{domain_name}/"
os.makedirs(data_output_dir, exist_ok=True)

ds_out = xr.Dataset(
    {
        "VHF_submeso_fullEOS": VHF_submeso_fullEOS,
        "VHF_steric_fullEOS": VHF_steric_fullEOS,
        "VHF_submeso_linearEOS": VHF_submeso_linear,
        "VHF_steric_linearEOS": VHF_steric_linear,
        "Q_eddy_daily": Q_eddy_daily,
    },
    attrs=dict(
        description="Domain-mean VHF theory using buoyancy ratio maps",
        domain=domain_name,
        Const=Const,
        units="W m-2",
    )
)

nc_path = os.path.join(
    data_output_dir,
    "VHF_timeseries_MapBuoyancyRatio.nc"
)

ds_out.to_netcdf(nc_path)
print(f"Saved NetCDF file to: {nc_path}")

# =====================
# Plot
# =====================
dates = pd.to_datetime(VHF_submeso_fullEOS.time.values)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dates, VHF_submeso_fullEOS,
        label="Submeso (Full EOS r)",
        linewidth=2)

ax.plot(dates, VHF_submeso_linear,
        linestyle="--",
        label="Submeso (Linear EOS r)",
        linewidth=2)

ax.plot(dates, VHF_steric_fullEOS,
        label="Steric (Full EOS r)",
        linewidth=2)

ax.plot(dates, VHF_steric_linear,
        linestyle="--",
        label="Steric (Linear EOS r)",
        linewidth=2)

ax.plot(dates, Q_eddy_daily,
        color="k",
        linewidth=2.5,
        label="Diagnosed Eddy Heat Flux")

ax.axhline(0, color="k", lw=0.8)
ax.set_title("Vertical Heat Flux: Theory (Map r) vs Diagnosed Eddy Flux")
ax.set_ylabel("VHF (W m$^{-2}$)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

plt.tight_layout()

fig_path = os.path.join(figdir, "VHF_theory_MapBuoyancyRatio.png")
fig.savefig(fig_path, dpi=300)

print(f"Saved figure to: {fig_path}")
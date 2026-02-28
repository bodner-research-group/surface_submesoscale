import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.io import savemat  # <-- For saving .mat files

# from set_constant import domain_name, face, i, j
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain


from set_colormaps import WhiteBlueGreenYellowRed
import cmocean

cmap = WhiteBlueGreenYellowRed()
plt.rcParams.update({'font.size': 16})

output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/figs/{domain_name}"

###############
##### Save the following time series as .mat data
###############

# --- Load datasets ---

### 1.1 Net surface buoyancy flux
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day_Bflux.nc"
Bflux_daily_avg = xr.open_dataset(fname).Bflux_daily_avg

### 1.2 Mixed-layer depth
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily_surface_reference_GSW.nc"
Hml_mean = abs(xr.open_dataset(fname).Hml_mean)


### 1.3 Mean stratification of 50%-90% of the mixed layer
# fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_7d_rolling.nc" # N2ml_weekly.nc
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily_surface_reference_GSW.nc"
N2ml_mean = xr.open_dataset(fname).N2ml_mean

### 2.1 Turner Angle agreement (% of casts with |TuV-TuH|<= 10 degrees)
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_Timeseries_Stats_7d_rolling.nc"
Tu_agreements_pct = xr.open_dataset(fname).Tu_agreements_pct
Tu_diff_means = xr.open_dataset(fname).Tu_diff_means


### 2.2 SSH gradient magnitude
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_gradient/SSH_gradient_magnitude.nc"
eta_grad_mag_weekly = xr.open_dataset(fname).eta_grad_mag_weekly


### 3.1 Wavelength of the most unstable mixed-layer-instability (MLI) waves
# fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_7d_rolling.nc"
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily_surface_reference_GSW.nc"
Lambda_MLI_mean = xr.open_dataset(fname).Lambda_MLI_mean/1000  # in km

### 3.2 Surface energy injection scale (TO DO)

### 3.4 Wavelength corresponding to the peak in the isotropic spectra of SSH anomaly (much larger than submesoscale)


### Wind stress
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux/windstress_center_daily_avg.nc"
ds = xr.open_dataset(fname)

taux = ds["taux_center"]
tauy = ds["tauy_center"]

# Wind-stress magnitude
print("Computing wind-stress magnitude...")
tau_mag = np.sqrt(taux**2 + tauy**2)

# Spatial average (if desired)
tau_mag_mean = tau_mag.mean(dim=("i", "j"))



# --- Prepare dictionary to save ---
mat_data = {
    "Bflux_daily_avg": Bflux_daily_avg.values,
    "Hml_mean": Hml_mean.values,
    "N2ml_mean": N2ml_mean.values,
    "Tu_agreements_pct": Tu_agreements_pct.values,
    "Tu_diff_means": Tu_diff_means.values,
    "eta_grad_mag_weekly": eta_grad_mag_weekly.values,
    "Lambda_MLI_mean": Lambda_MLI_mean.values,
    "tau_mag_mean":tau_mag_mean.values,
    # Optional: Save coordinates (e.g. time) if needed
    "time_B0": Bflux_daily_avg.time.values.astype('datetime64[s]').astype(str),
    "time_N2": N2ml_mean.time.values.astype('datetime64[s]').astype(str),
    "time_Hml": Hml_mean.time.values.astype('datetime64[s]').astype(str),
    "time_Tu": Tu_diff_means.date.values.astype('datetime64[s]').astype(str),
    "time_lambda": Lambda_MLI_mean.time.values.astype('datetime64[s]').astype(str),
    "time_wind": tau_mag_mean.time.values.astype('datetime64[s]').astype(str),
    # Add others as needed
}

# --- Save to .mat file ---
savemat(f"{output_dir}/timeseries_data_new_GSW.mat", mat_data)

print("Saved to timeseries_data.mat")
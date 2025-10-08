import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.io import savemat  # <-- For saving .mat files

from set_constant import domain_name, face, i, j
from set_colormaps import WhiteBlueGreenYellowRed
import cmocean

cmap = WhiteBlueGreenYellowRed()
plt.rcParams.update({'font.size': 16})

output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}"

###############
##### Save the following time series as .mat data
###############

# --- Load datasets ---

### 1.1 Net surface buoyancy flux
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day.nc"
qnet_7day_smooth = xr.open_dataset(fname).qnet_7day_smooth

### 1.2 Mixed-layer depth
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries.nc" # Hml_weekly_mean.nc
Hml_mean = xr.open_dataset(fname).Hml_mean

### 1.3 Mean stratification of 30%-90% of the mixed layer
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries.nc" # N2ml_weekly.nc
N2ml_mean = xr.open_dataset(fname).N2ml_mean


### 2.1 Turner Angle agreement (% of casts with |TuV-TuH|<= 10 degrees)
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_Timeseries_Stats.nc"
Tu_agreements_pct = xr.open_dataset(fname).Tu_agreements_pct
Tu_diff_means = xr.open_dataset(fname).Tu_diff_means


### 2.2 SSH gradient magnitude
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_gradient/SSH_gradient_magnitude.nc"
eta_grad_mag_weekly = xr.open_dataset(fname).eta_grad_mag_weekly

### 2.3 Averaged wb cross-spectra within the mixed layer and within the submesoscale range (TO DO)
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_max_spec_vp_filtered.nc"
mean_spec_in_MLD_submeso = xr.open_dataset(fname).mean_spec_in_MLD_submeso
mean_spec_in_MLD = xr.open_dataset(fname).mean_spec_in_MLD

### (*?) 2.4 Mean SSH gradient magnitude for regions with |TuV-TuH|<= 10 degrees (TO DO)



### 3.1 Wavelength of the most unstable mixed-layer-instability (MLI) waves
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries.nc"
Lambda_MLI_mean = xr.open_dataset(fname).Lambda_MLI_mean/1000  # in km

### 3.2 Surface energy injection scale (TO DO)

### 3.3 Wavelength corresponding to the peak in the isotropic spectra of SSH anomaly (TO DO)

### 3.4 Wavelength at the peak of the wb cross-spectra
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_max_spec_vp_filtered.nc"
Lr_at_max = xr.open_dataset(fname).Lr_at_max  # in km



# --- Prepare dictionary to save ---
mat_data = {
    "qnet_7day_smooth": qnet_7day_smooth.values,
    "Hml_mean": Hml_mean.values,
    "N2ml_mean": N2ml_mean.values,
    "Tu_agreements_pct": Tu_agreements_pct.values,
    "Tu_diff_means": Tu_diff_means.values,
    "eta_grad_mag_weekly": eta_grad_mag_weekly.values,
    "Lambda_MLI_mean": Lambda_MLI_mean.values,
    "Lr_at_max": Lr_at_max.values,
    "mean_spec_in_MLD": mean_spec_in_MLD.values,
    "mean_spec_in_MLD_submeso": mean_spec_in_MLD_submeso.values,
    # Optional: Save coordinates (e.g. time) if needed
    "time_qnet": qnet_7day_smooth.time.values.astype('datetime64[s]').astype(str),
    "time_Hml": Hml_mean.time.values.astype('datetime64[s]').astype(str),
    # Add others as needed
}

# --- Save to .mat file ---
savemat(f"{output_dir}/timeseries_data.mat", mat_data)

print("Saved to timeseries_data.mat")
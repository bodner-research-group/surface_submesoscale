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


output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/Manuscript_Data/{domain_name}"
os.makedirs(output_dir, exist_ok=True)


###############
##### Save the following time series as .mat data
###############

# --- Load datasets ---

###### Total SSH gradient magnitude
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_gradient/SSH_gradient_magnitude.nc"
eta_grad_mag_daily = xr.open_dataset(fname).eta_grad_mag_daily

###### Steric height gradient magnitude
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly_timeseries_surface_reference/grad2_timeseries.nc"
eta_steric_grad_mean = xr.open_dataset(fname).eta_prime_grad_mean
eta_grad_mean = xr.open_dataset(fname).eta_grad_mean

###### Submesoscale SSH gradient magnitude
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_Gaussian_submeso_LambdaMLI_timeseries.nc"
eta_submeso_grad_mean = xr.open_dataset(fname).eta_submeso_grad_mean


fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_Gaussian_submeso_17kmCutoff_timeseries.nc"
eta_submeso_grad_mean_winter = xr.open_dataset(fname).eta_submeso_grad_mean



###### Wavelength of the most unstable mixed-layer-instability (MLI) waves
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily_surface_reference.nc"
Lambda_MLI_mean = xr.open_dataset(fname).Lambda_MLI_mean/1000  # in km


# --- Prepare dictionary to save ---
mat_data = {
    "eta_grad_mag_daily":eta_grad_mag_daily.values,
    "eta_grad_mean":eta_grad_mean.values,
    "eta_steric_grad_mean":eta_steric_grad_mean.values,
    "eta_submeso_grad_mean":eta_submeso_grad_mean.values,
    "eta_submeso_grad_mean_winter":eta_submeso_grad_mean_winter.values,
    "Lambda_MLI_mean": Lambda_MLI_mean.values,
    # Optional: Save coordinates (e.g. time) if needed
    "time_SSH": eta_grad_mag_daily.time.values.astype('datetime64[s]').astype(str),
    "time_SSH_mean": eta_grad_mean.time.values.astype('datetime64[s]').astype(str),
    "time_steric": eta_steric_grad_mean.time.values.astype('datetime64[s]').astype(str),
    "time_submesoSSH": eta_submeso_grad_mean.time.values.astype('datetime64[s]').astype(str),
    "time_lambda": Lambda_MLI_mean.time.values.astype('datetime64[s]').astype(str),
    # Add others as needed
}

# --- Save to .mat file ---
savemat(f"{output_dir}/timeseries_steric_submesoSSH.mat", mat_data)

print("Saved to timeseries_steric_submesoSSH.mat")
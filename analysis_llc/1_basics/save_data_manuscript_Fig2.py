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


###### Steric height


###### Submesoscale SSH
# maps
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_Gaussian_submeso_LambdaMLI.nc"
SSH_submesoscale_map = xr.open_dataset(fname).SSH_submesoscale.isel(time=61+45)
# timeseries
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_Gaussian_submeso_LambdaMLI_timeseries.nc"
eta_submeso_grad_mean = xr.open_dataset(fname).eta_submeso_grad_mean





# --- Prepare dictionary to save ---
mat_data = {
    "SSH_submesoscale_map": SSH_submesoscale_map.values,
    "eta_submeso_grad_mean":eta_submeso_grad_mean.values,
    # Optional: Save coordinates (e.g. time) if needed
    "time_submesoSSH": eta_submeso_grad_mean.time.values.astype('datetime64[s]').astype(str),
    # Add others as needed
}

# --- Save to .mat file ---
savemat(f"{output_dir}/timeseries_steric_submesoSSH.mat", mat_data)

print("Saved to timeseries_steric_submesoSSH.mat")
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

### Net surface buoyancy flux
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day_Bflux.nc"
Bflux_daily_avg = xr.open_dataset(fname).Bflux_daily_avg

### 1.2 Mixed-layer depth
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_daily_surface_reference.nc"
Hml_mean = abs(xr.open_dataset(fname).Hml_mean)

# ### 1.3 Mean stratification of 30%-90% of the mixed layer
# fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Lambda_MLI_timeseries_7d_rolling.nc" # N2ml_weekly.nc
# N2ml_mean = xr.open_dataset(fname).N2ml_mean

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
    "tau_mag_mean":tau_mag_mean.values,
    # Optional: Save coordinates (e.g. time) if needed
    "time_B0": Bflux_daily_avg.time.values.astype('datetime64[s]').astype(str),
    "time_Hml": Hml_mean.time.values.astype('datetime64[s]').astype(str),
    "time_wind": tau_mag_mean.time.values.astype('datetime64[s]').astype(str),
    # Add others as needed
}

# --- Save to .mat file ---
savemat(f"{output_dir}/timeseries_surface_forcing.mat", mat_data)

print("Saved to timeseries_data.mat")
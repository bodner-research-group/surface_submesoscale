import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from set_constant import domain_name, face, i, j
from set_colormaps import WhiteBlueGreenYellowRed
import cmocean

cmap = WhiteBlueGreenYellowRed()
plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

###############
##### Save the following time series as .mat data
###############

### 1.1 Net surface buoyancy flux
file1_1 = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day.nc"
qnet_7day_smooth = xr.open_dataset(file1_1).qnet_7day_smooth

### 1.2 Mixed-layer depth
file1_2 = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Hml_weekly_mean.nc"
Hml_mean = xr.open_dataset(file1_2).Hml_mean

### 1.3 Mean stratification of 50%-90% of the mixed layer
file1_3 = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/", "N2ml_weekly.nc")
N2ml_mean = xr.open_dataset(file1_3).N2ml_mean


### 2.1 Wavelength of the most unstable mixed-layer-instability (MLI) waves

### 2.2 Wavelength at the peak of the wb cross-spectra

### 2.3 Surface energy injection scale (TO DO)

### 2.4 Wavelength corresponding to the peak in the isotropic spectra of SSH anomaly (TO DO)



### 3.1 Turner Angle agreement (% of casts with |TuV-TuH|<= 10 degrees)

### 3.2 Averaged wb cross-spectra within the mixed layer and within the submesoscale range

### 3.3 SSH gradient magnitude

### 3.4 Mean SSH gradient magnitude for regions with |TuV-TuH|<= 10 degrees



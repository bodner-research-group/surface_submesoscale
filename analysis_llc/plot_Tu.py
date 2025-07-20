# Load packages
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr 
import dask 
import gsw  # (Gibbs Seawater Oceanography Toolkit) https://teos-10.github.io/GSW-Python/gsw.html
from numpy.linalg import lstsq
import seaborn as sns
from scipy.stats import gaussian_kde

# Folder to store the figures
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_day1_3pm"

# Open the NETCDF file
ds_out = xr.open_dataset(f"{figdir}/Tu_difference.nc")
ds_out.load()

# Load all variables
locals().update({k: ds_out[k] for k in ds_out.data_vars})
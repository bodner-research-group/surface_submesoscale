# Code Description

[To be completed]

## analysis_llc

- **set_constant.py**  
  Set domain name, horizontal indices, and the start and end time indices.

- **py01_24h_avg_TSW.py**  
  Compute daily mean temperature, salinity, and vertical velocity.

- **py03_rho_daily.py**  
  Compute daily mean potential density and 10-m referenced mixed layer depth.

- **py03_Hml_daily.py**  
  Compute surface (0.5-m) referenced mixed layer depth.

- **py31_wb_CoarseGraining_MixedlayerAvg_calc.py, py31_wb_CoarseGraining_MixedlayerAvg_timeseries.py**  
  Compute total, mean, and eddy vertical buoyancy fluxes using daily averaged data. Compute their domain averages, and plot the time series.

- **py31_wb_CoarseGraining_MixedlayerAvg_calc_hourly.py, py31_wb_CoarseGraining_MixedlayerAvg_timeseries.py**  
  Compute total, mean, and eddy vertical buoyancy fluxes using hourly model output. 

- **py23_rho_insitu_hydrostatic_pressure.py**  
  Compute in-situ density and estimate hydrostatic pressure.

- **py24_steric_height_Wang25_timeseries.py**  
  Compute and save the domain-averaged time series of steric height anomaly following Jinbo Wang et al. (2025).

- **py28_Ekman_buoyancy_flux_hourly.py, py28_Ekman_buoyancy_flux_daily.py**  
  Compute Ekman buoyancy flux using hourly or daily model output.

- **py25_SSH_submesoscale_GaussianFilter.py**  
  Compute submesoscale sea surface height (SSH) gradient by applying a Gaussian filter to SSH.

- **py25_timeseries_SSH_submeso_grad_laplace.py**  
  Compute the time series of domain-averaged submesoscale SSH gradient magnitude.


## analysis_swot

- **llc2swot_sbatch.py**  
  Extracts sea surface height anomaly data from LLC4320 for the same day of the year, but a different year than the SWOT observations. This script was adapted from `model2SWOT.py` in the [SynthOcean](https://github.com/Amine-ouhechou/synthocean) package and includes parallel computing.

- **llc2swot_combine_nc.py**  
  Combines all individual swaths from LLC4320 into a single NetCDF file.

- **llc2swot_plot.py**  
  Generates plots of the LLC4320 swaths.

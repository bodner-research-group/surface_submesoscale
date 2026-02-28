# Data and Code Description

## Manuscript_Data
  All data used to generate the figures in the manuscript and Supporting Information.

## Manuscript_Figures
  MATLAB scripts for generating all figures in the manuscript and Supporting Information.

## analysis_llc
### 1_basics:
- **script1_find_indices.py**
  Find the i, j, and face indices for selected region.

- **set_constant.py, set_colormaps.py**  
  Set domain name, horizontal indices, the start and end time indices, and colormaps.

- **py01_24h_avg_TSW.py, py02_24h_avg_surface.py**  
  Compute daily mean 3D output: temperature, salinity, and vertical velocity.
  Compute daily mean 2D output: surface velocities, wind stress, SSH, KPP boundary layer depth, etc.

- **py03_rho_daily.py, py03_Hml_daily.py**  
  Compute daily mean potential density and 10-m referenced mixed layer depth.
  Compute surface (0.5-m) referenced mixed layer depth.

- **py08_vorticity_strain_calc.py, py08_vorticity_strain_plot.py**
  Compute and plot surface strain rate, surface vorticity, and surface divergence

- **py19_Lambda_MLI.py, py20_Lambda_MLI_timeseries_excludeShallowHml,py, py20_Lambda_MLI_timeseries.py**
  Compute mixed layer horizontal buoyancy gradient M2, vertical buoyancy gradient N2, and the wavelength of the most unstable mixed layer instability waves.

- **save_data_manuscript_Fig2.py, save_data_manuscript_surface.py, save_data_timeseries_Fig1.py**
  Save timeseries and spatial fields needed for creating figures for the manuscript.

### 2_surface_forcing:
- **py28_Ekman_buoyancy_flux_hourly.py, py28_Ekman_buoyancy_flux_daily.py**  
  Compute Ekman buoyancy flux using hourly or daily model output.

- **py07_Qnet_mean.py**  
  Compute surface buoyancy flux (contributed by net heat flux/freshwater flux)

- **py27_wind_stress_daily.py**  
  Calculate daily and domain-averaged wind stress magnitude.

- **py29_KPPhbl_vs_MLD.py**  
  Plot the KPP boundary layer height and compare it to the mixed layer depth.


### 3_eddy_buoyancy_flux:
- **py31_hourly_rho.py**  
  Compute hourly potential density, surface referenced mixed layer depth, and 10-m referenced mixed layer depth. 

- **hourly_wb_Gaussian_calc.py (parallel), hourly_wb_Gaussian_calc_sequential.py, py31_hourly_wb_timeseries.py (parallel), py31_hourly_wb_timeseries_sequential.py**
  Compute total, mean, and eddy vertical buoyancy fluxes using hourly model output and a Gaussian filter, exclude regions where the mixed layer depth is shallower than 10m. The filter size is a constant (e.g., 30 km), or determined from the 60-day average wavelength of the most unstable mode of mixed-layer instability.

- **py31_hourly_wb_timeseries.py**
  Compute the domain averaged eddy vertical buoyancy fluxes, and plot the time series.  

- **py31_hourly_wb_calc.py (parallel), py31_hourly_wb_calc_sequential.py**
  Compute total, mean, and eddy vertical buoyancy fluxes using hourly model output and coarse-graining ("box filter"), exclude regions where the mixed layer depth is shallower than 10m.

- **py31_wb_CoarseGraining_MixedlayerAvg_calc.py, py31_wb_CoarseGraining_MixedlayerAvg_plot.py, py31_wb_CoarseGraining_MixedlayerAvg_timeseries.py**  
  Compute total, mean, and eddy vertical buoyancy fluxes using daily averaged data, using a box filter. Compute their domain averages, and plot the time series.


### 4_steric_height:
- **py23_rho_insitu_hydrostatic_pressure.py**  
  Compute in-situ density and estimate hydrostatic pressure.

- **py24_steric_height_Wang25_timeseries.py, py24_steric_height_Wang25_timeseries2.py**
  Compute and save the domain-averaged time series of steric height anomaly following Jinbo Wang et al. (2025).

- **save_steric_gradient_maps.py**
  Save maps of the gradient magnitude of the steric height for the manuscript.


### 5_SSH:
- **py25_SSH_submesoscale_GaussianFilter_time_varying.py**  
  Compute submesoscale sea surface height (SSH) gradient by applying a time-varying Gaussian filter to SSH. The filter scale is set by the wavelength of the most unstable mixed layer instability waves.

- **py25_SSH_submesoscale_GaussianFilter.py**  
  Compute submesoscale sea surface height (SSH) gradient by applying a constant Gaussian filter to SSH.

- **py25_timeseries_SSH_submeso_grad_laplace.py**  
  Compute the time series of domain-averaged submesoscale SSH gradient magnitude.

- **save_SSH_submeso_gradient_maps.py**
  Save the data required to calculate the vertical heat flux.


### 6_Turner_angle:
- **TurnerAngle_BuoyancyRatio_daily_calc.py, TurnerAngle_BuoyancyRatio_timeseries.py**  
  Compute the horizontal and vertical Turner angles and the density ratio. Save the data and plots, and compute domain-averaged time series. 

### 7_mixed_layer_depth:
- **py30_Hml_tendency_ver3.py**
  Plot the daily-averaged budget terms driving the change of the mixed layer depth; plot a 7-day rolling mean.
  Reconstruct mixed layer depth from these terms, and compare it against the true mixed layer depth.
  
### 8_vertical_heat_flux:
- **hourly_VHF_diagnosed_calc.py (parallel), hourly_VHF_diagnosed_calc_sequential.py, hourly_VHF_timeseries.py**
  Compute total, mean, and eddy vertical heat fluxes using hourly model output and a Gaussian filter, exclude regions where the mixed layer depth is shallower than 10m. The filter size is a constant (e.g., 30 km), or determined from the 60-day average wavelength of the most unstable mode of mixed-layer instability.

- **VHT_theory_BuoyancyRatio.py**
  The vertical heat flux is reconstructed from the surface buoyancy ratio and mixed-layer steric height or submesoscale SSH, and then compared with the diagnosed VHF.

- **VHF_theory_calc.py**
  The vertical heat flux is reconstructed from the surface horizontal Turner angle and mixed-layer steric height or submesoscale SSH, then compared with the diagnosed VHF.



## analysis_swot

- **llc2swot_sbatch.py**  
  Extracts sea surface height anomaly data from LLC4320 for the same day of the year, but a different year than the SWOT observations. This script was adapted from `model2SWOT.py` in the [SynthOcean](https://github.com/Amine-ouhechou/synthocean) package and includes parallel computing.

- **llc2swot_combine_nc.py**  
  Combines all individual swaths from LLC4320 into a single NetCDF file.

- **llc2swot_plot.py**  
  Generates plots of the LLC4320 swaths.

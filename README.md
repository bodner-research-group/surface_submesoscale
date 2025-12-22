# Code Description

[To be completed]

## analysis_llc

- **py32_Ekman_buoyancy_flux_hourly.py**  
  Compute hourly Ekman buoyancy flux.

- **py25_SSH_submesoscale_GaussianFilter.py**  
  Compute submesoscale sea surface height (SSH) gradient by applying a Gaussian filter to SSH.

- **py25_timeseries_SSH_submeso_grad_laplace.py**  
  Compute the time series of domain-averaged submesoscale SSH gradient magnitude.

## analysis_swot

- **llc2swot_sbatch.py**  
  Extracts sea surface height anomaly data from LLC4320 for the same day of the year, but a different year than the SWOT observations.  
  This script was adapted from `model2SWOT.py` in the [SynthOcean](https://github.com/Amine-ouhechou/synthocean) package and includes parallel computing.

- **llc2swot_combine_nc.py**  
  Combines all individual swaths from LLC4320 into a single NetCDF file.

- **llc2swot_plot.py**  
  Generates plots of the LLC4320 swaths.

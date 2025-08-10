##### Time series of the following variables:
#####
##### Qnet (net surface heat flux into the ocean),
##### Hml (mixed-layer depth), 
##### TuH (horizontal Turner angle), 
##### TuV (vertical Turner angle),
##### wb_cros (variance-perserving cross-spectrum of vertical velocity and buoyancy)
##### wbmin (the minimum of wb_cros)
##### Lmax (the horizontal length scale corresponds to wbmin), 
##### Dmax (the depth corresponds to wbmin), 
##### gradSSH (absolute gradient of sea surface height anomaly), etc.
#####
##### Step 1: compute 24-hour averages of temperature, salinity, and vertical velocity, save as .nc files
##### Step 2: compute 7-day averages of potential density, alpha, beta, Hml, save as .nc files
##### Step 3: compute wb_cros using the 24-hour averages, and then compute the 7-day averaged wb_cros 
##### Step 4: plot wb_cros of each week, compute wbmin, Lmax, Dmax
##### Step 5: compute 7-day movmean of Qnet
##### Step 6: compute TuH and TuV



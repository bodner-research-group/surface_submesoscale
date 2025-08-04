##### Time series of the following variables:
#####
##### Qnet (net surface heat flux into the ocean),
##### Hml (mixed-layer depth), 
##### TuH (horizontal Turner angle), 
##### TuV (vertical Turner angle),
##### wb_cros (variance-perserving cross-spectrum of vertical velocity and buoyancy), 
##### Lmax (the horizontal length scale corresponds to wb_cros minimum), 
##### Dmax (the depth corresponds to wb_cros minimum), 
##### gradSSH (absolute gradient of sea surface height anomaly), etc.
#####
##### Step 1: compute 12-hour averages of temperature, salinity, and vertical velocity, save as .nc files
##### Step 2: compute 7-day averages of potential density, alpha, beta, Hml, save as .nc files
##### Step 3: compute wb_cros using the 12-hour averages, and then compute the 7-day averaged wb_cros using a sliding window
##### Step 4: compute TuH and TuV using the 7-day averaged data 



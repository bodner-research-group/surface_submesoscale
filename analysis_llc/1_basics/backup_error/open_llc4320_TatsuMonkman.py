# Script created by Tatsu Monkman
# Load packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr # https://docs.xarray.dev/en/stable/index.html
import zarr # https://zarr.dev/
import dask # https://www.dask.org/

# Open zarr store with xarray
# This opens the LLC4320 data as an xarray dataset with predefined dask chunks (if dask is imported)
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Call the dataset to look at what variables, dimensions, and coordinates are present.
# An xarray "Data Set" is basically a collection of numpy arrays that have
# shared coordinates and dimensions. See https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
# (Call the line below in individual jupyter notebook cell)
ds1

# Call a variable from the dataset to look at dask chunks. This
# returns an xarray "Data Array" (basically a single numpy array with labeled dims and coordinates).
# See https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html 
# and https://examples.dask.org/xarray.html
# (Call the line below in individual jupyter notebook cell)
ds1.Theta

# Both MITgcm grid variables and output variables are
# stored as "variables" in the xarray dataset. For example,
# to get the horizontal cell area (rA) you can just call:
# (Call the line below in individual jupyter notebook cell)
ds1.rA

# If you only want a couple of variables, you can call them by 
# loading them from the dataset using a list:
# (Call the line below in individual jupyter notebook cell)
ds1[["Theta","U"]]

# Call the "time" dimension on ds1.Theta to see "coordinates"
# associated with the time dimension. Coordinates are a useful but
# sometimes tricky feature of xarray. See https://docs.xarray.dev/en/latest/generated/xarray.Coordinates.html
# (Call the line below in individual jupyter notebook cell)
ds1.Theta.time

# You can subset xarray datasets very easily. For example, to 
# subset Theta to get the data at time=0, k=0, and face=2 and look
# at the resulting data you can just do the following:
# (NOTE: the values time=0, k=0, face=2 are the indicies at which I'm calling the data. 
# Think of it like a numpy array, where you call individual values from the numpy ndarray using their 
# Python index, i.e. np.asarray([0,1,2,3])[2] = 2). There are more advanced 
# methods for subsetting by coordinate values and data values (for example, NaNs)
# (Call the two lines below in individual jupyter notebook cell)
ds1_theta_t0_k0_face2 = ds1.Theta.isel(time=0,k=0,face=2)
ds1_theta_t0_k0_face2

# You can also subset the whole dataset at once! This will select the values
# of variables only at the location you are interested in. An example of the usefullness of xarray
# (Call the two lines below in individual jupyter notebook cell)
ds1_t0_k0_face2 = ds1.isel(time=0,k=0,face=2)
ds1_t0_k0_face2

# You can call standard array operations on xarray dataarrays fairly
# intuitively:
# (Call the two lines below in individual jupyter notebook cell)
diff_Theta_t0_t1 = ds1.Theta.isel(time=0,k=0,face=2,j=slice(0,4320,5),i=slice(0,4320,5)) - ds1.Theta.isel(time=24,k=0,face=2,j=slice(0,4320,5),i=slice(0,4320,5))
diff_Theta_t0_t1.plot(vmin=-.3,vmax=.3,cmap="seismic")

# xarray has tons of built in functions to take means, standardeviations, etc built in.
# For example, you can take means along some specific dimensions you are interested in like so:
# (Call the two lines below in individual jupyter notebook cell)
ds1_Theta_t0_k0_face2_t_mean = ds1.Theta.isel(time=slice(0,100),k=0,face=2).mean(dim="time")
ds1_Theta_t0_k0_face2_t_mean.plot()

# When used with dask and zarr, xarray will do "lazy" computations.
# This means that a given computation isn't actually done until
# you explicitly call the values of the data for, say, plotting. For example, to see what
# the mean value is in "i" below you can call the underlying numpy array
# that xarray is supposed to calculate. You can do this just by calling
# "XARRAY_DATAARRAY.values":
# (Call the two lines below in individual jupyter notebook cell)
ds1_Theta_t0_k0_face2_i_mean = ds1.Theta.isel(time=0,k=0,face=2).mean(dim="i")
ds1_Theta_t0_k0_face2_i_mean.values

# Plot some temperature data (if you are using jupyter notebook you can 
# also do this for the first timestep just by calling "ds1.Theta.isel(time=0,k=0,face=2,j=slice(0,4320,5),i=slice(0,4320,5)).plot()" in 
# a single cell). NOTE: I'm only taking every 5th value in the i- and j-directions. This is just for faster plotting
fig, axs = plt.subplots(1,2,figsize=(15,5))
ds1.Theta.isel(time=0,k=0,face=2,j=slice(0,4320,5),i=slice(0,4320,5)).plot(ax=axs[0],vmin=-10,vmax=30)
ds1.Theta.isel(time=5000,k=0,face=2,j=slice(0,4320,5),i=slice(0,4320,5)).plot(ax=axs[1],vmin=-10,vmax=30)
plt.show()
plt.close()

# Plot some data! Here I am plotting the Theta values at k=0 and face=2 for times=[0, 20, 40, 60, 80, 100]
# and j = (0,5,10,...,4310,4315) and i = (0,5,10,...,4310,4315):
time=0
face=2
k = 0
j = slice(0,4320,5)
i = slice(0,4320,5)
for time in range(0,100,20):
    plt.figure(figsize=(20,12),dpi=200)
    ds1.Theta.isel(time=time,k=k,face=face,j=j,i=i).plot(cmap="Spectral_r",vmin=2,vmax=25)
    plt.title(f"time={ds1.Theta.time[time].values.astype('datetime64[m]')}, face={face}, Depth={np.around(ds1.Z.values[k])}m")
    #plt.savefig(f"./movie_figs/face02_time_k13/face{str(face).zfill(2)}_time{str(time).zfill(6)}_k{k}.png")
    plt.show()
    plt.close()
	
# Here is another plotting script for a subset
time=0
face=2
k = 13
j = slice(1000,2000)
i = slice(0,1000)
for time in range(0,100,20):
    plt.figure(figsize=(20,12),dpi=200)
    ds1.Theta.isel(time=time,k=k,face=face,j=j,i=i).plot(cmap="Spectral_r",vmin=15,vmax=25)
    plt.title(f"time={ds1.Theta.time[time].values.astype('datetime64[m]')}, face={face}, Depth={np.around(ds1.Z.values[k])}m")
    #plt.savefig(f"./movie_figs/face02_time_k13/face{str(face).zfill(2)}_time{str(time).zfill(6)}_k{k}.png")
    plt.show()
    plt.close()
    

# Good luck!
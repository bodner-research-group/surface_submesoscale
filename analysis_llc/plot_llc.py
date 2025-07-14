# Plot T, S, U, V of the LLC4320 data
# Load packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr 
import dask 
import multiprocessing as mp


# Read the data
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)
list(ds1.data_vars)

figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs"

face = 1
k = 0
# time = slice(0,10311,20)
i = slice(0,4320,2)
j = slice(0,4320,2)

# tt = ds1.Theta.isel(time=time,k=k,face=face,i=i,j=j)
# ss = ds1.Salt.isel(time=time,k=k,face=face,i=i,j=j)
# uu = ds1.U.isel(time=time,k=k,face=face,i_g=i,j=j)
# vv = ds1.V.isel(time=time,k=k,face=face,i=i,j_g=j)
# eta = ds1.Eta.isel(time=time,face=face,i=i,j=j)

# lon = ds1.XC.isel(face=face,i=i,j=j)
# lat = ds1.YC.isel(face=face,i=i,j=j).transpose('j', 'i')
# lon_g = ds1.XG.isel(face=face,i_g=i,j_g=j)
# lat_g = ds1.YG.isel(face=face,i_g=i,j_g=j).transpose('j_g', 'i_g')

# Plots
for time in range(0,5000,5):
    plt.figure(figsize=(20,12),dpi=300)
    ds1.Theta.isel(time=time,k=k,face=face,j=j,i=i).plot(cmap="Spectral_r",vmin=-2,vmax=30)
    plt.title(f"time={ds1.Theta.time[time].values.astype('datetime64[m]')}, face={face}, Depth={np.around(ds1.Z.values[k])}m")
    plt.savefig(f"{figdir}/face01_time_k0/tt_face{str(face).zfill(2)}_time{str(time).zfill(6)}_k{k}.png")
    plt.show()
    plt.close()


# Plots
for time in range(0,5000,5):
    plt.figure(figsize=(20,12),dpi=300)
    ds1.U.isel(time=time,k=k,face=face,i_g=i,j=j).plot(cmap="coolwarm",vmin=-1,vmax=2)
    plt.title(f"time={ds1.U.time[time].values.astype('datetime64[m]')}, face={face}, Depth={np.around(ds1.Z.values[k])}m")
    plt.savefig(f"{figdir}/face01_time_k0/uu_face{str(face).zfill(2)}_time{str(time).zfill(6)}_k{k}.png")
    plt.show()
    plt.close()


# # Plot surface velocity
# surface_u = ds1.U.isel(time=0,k=k,face=face,i_g=i,j=j) 
# surface_u.plot(vmin=-1,vmax=1,cmap="coolwarm")
# plt.savefig(f"{figdir}/surface_u.png", dpi=150)
# plt.close()

# # Plot surface temperature
# surface_T = ds1.Theta.isel(time=0,k=k,face=face,i=i,j=j) 
# surface_T.plot(vmin=2,vmax=25,cmap="coolwarm")
# plt.savefig(f"{figdir}/surface_T.png", dpi=150)
# plt.close()


# ds1_Theta_t0_k0_face2_t_mean = ds1.Theta.isel(time=slice(0,100),k=0,face=2).mean(dim="time")
# ds1_Theta_t0_k0_face2_t_mean.plot()

# ds1_Theta_t0_k0_face2_i_mean = ds1.Theta.isel(time=0,k=0,face=2).mean(dim="i")
# ds1_Theta_t0_k0_face2_i_mean.values

# fig, axs = plt.subplots(1,2,figsize=(15,5))
# ds1.Theta.isel(time=0,k=0,face=2,j=slice(0,4320,5),i=slice(0,4320,5)).plot(ax=axs[0],vmin=-10,vmax=30)
# ds1.Theta.isel(time=5000,k=0,face=2,j=slice(0,4320,5),i=slice(0,4320,5)).plot(ax=axs[1],vmin=-10,vmax=30)

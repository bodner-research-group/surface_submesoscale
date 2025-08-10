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

# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os
from dask.distributed import Client, LocalCluster
from glob import glob
import time
import xrft
from dask import delayed, compute
from numpy.linalg import lstsq

# ========== Setup Dask distributed cluster ==========
cluster = LocalCluster(n_workers=16, threads_per_worker=1, memory_limit="20GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Define input and output directories ==========
input_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin"
output_dir = os.path.join(input_dir, "wb_cross_spectra_weekly")
os.makedirs(output_dir, exist_ok=True)

# ========== Define spatial domain slices ==========
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ========== Get all weekly 24-hour average files, sorted by date ==========
tt_files = sorted(glob(os.path.join(input_dir, "tt_24h_*.nc")))
ss_files = sorted(glob(os.path.join(input_dir, "ss_24h_*.nc")))
ww_files = sorted(glob(os.path.join(input_dir, "ww_24h_*.nc")))

# ========== Load lat, lon, and depth coordinates once from the full dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
depth = ds1.Z
depth_kp1 = ds1.Zp1

# Broadcast lon, lat, and depth arrays to have compatible shapes for calculation
lon_b, lat_b = xr.broadcast(lon, lat)
depth3d, _, _ = xr.broadcast(depth, lon_b, lat_b)

# ========== Calculate mean horizontal grid spacing (meters) and combined spacing dr ==========
dx = ds1.dxF.isel(face=face,i=i,j=j).mean().values
dy = ds1.dyF.isel(face=face,i=i,j=j).mean().values
dr = np.sqrt(dx**2/2 + dy**2/2)

def compute_one_spec(tt, ss, ww):
    """
    Compute the isotropic cross-spectrum between vertical velocity (w) and buoyancy (b).

    Parameters:
    tt: Potential temperature DataArray [time, k, j, i]
    ss: Practical salinity DataArray [time, k, j, i]
    ww: Vertical velocity DataArray [time, k_p1, j, i]

    Returns:
    wb_cross_spec: Cross-spectrum DataArray [freq_r, Z]
    """

    # compute buoyancy
    SA = gsw.SA_from_SP(ss, depth3d, lon_b, lat_b)
    CT = gsw.CT_from_pt(SA, tt)
    p_ref = 0
    rho0 = 1000
    gravity = 9.81
    rho = gsw.rho(SA, CT, p_ref)
    buoy = -gravity * (rho - rho0) / rho0

    # Rename k_p1 dimension to 'Z' (or any name for vertical coordinate)
    ww_z = ww.rename({'k_p1': 'Z'})

    # Assign depth coordinates to this vertical dimension
    ww_z = ww_z.assign_coords(Z=depth_kp1.values)

    # Interpolate ww_z from 'Z' to 'depth' (which corresponds to k layers)
    ww_interp = ww_z.interp(Z=depth)

    # Fill missing data with zeros to avoid NaNs in spectral calculations, rechunk
    buoy = buoy.fillna(0).chunk({'j': -1, 'i': -1})
    ww_interp = ww_interp.fillna(0).chunk({'j': -1, 'i': -1})

    # Compute isotropic cross-spectrum between vertical velocity and buoyancy
    spec = xrft.isotropic_cross_spectrum(
        ww_interp, buoy,
        dim=['i', 'j'],
        detrend='linear',
        window='hann',
        truncate=True
    ).compute()

    return spec

# ========== Loop through weekly 24-hour average files and compute cross-spectra ==========
for tt_file, ss_file, ww_file in zip(tt_files, ss_files, ww_files):
    date_str = os.path.basename(tt_file).split('_')[-1].replace('.nc','')
    out_file = os.path.join(output_dir, f"wb_cross_spec_vp_real_{date_str}.nc")
    if os.path.exists(out_file):
        print(f"Skipping {date_str}, already exists.")
        continue

    print(f"Processing {date_str}...")

    # Load datasets
    tt = xr.open_dataset(tt_file)['Theta']
    ss = xr.open_dataset(ss_file)['Salt']
    ww = xr.open_dataset(ww_file)['W']

    # Chunk time dimension
    tt = tt.chunk({'time': 1})
    ss = ss.chunk({'time': 1})
    ww = ww.chunk({'time': 1})

    # Create lists for delayed computation
    tasks = []
    for t in range(tt.sizes['time']):
        t_tt = tt.isel(time=t)
        t_ss = ss.isel(time=t)
        t_ww = ww.isel(time=t)
        task = delayed(compute_one_spec)(t_tt, t_ss, t_ww)
        tasks.append(task)

    # Run all and average
    results = compute(*tasks)
    WB_cross_spectra = xr.concat(results, dim='time').mean('time')

    # Compute radial wavenumber (cpkm) and variance-preserving spectrum
    k_r = WB_cross_spectra.freq_r / dr / 1e-3
    spec_vp = WB_cross_spectra * k_r

    # Save real part only
    ds_out = xr.Dataset({
        'spec_vp_real': spec_vp.real,
        'k_r': k_r,
        'depth': WB_cross_spectra.Z
    })

    ds_out.to_netcdf(out_file)
    print(f"Saved to {out_file}")
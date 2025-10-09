# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os
from dask.distributed import Client, LocalCluster
from glob import glob
import time
import xrft
import gc
from dask import delayed, compute

from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# ========== Define input and output directories ==========
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"
output_dir = os.path.join(f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}", "wb_cross_spectra_weekly")
os.makedirs(output_dir, exist_ok=True)

# ========== Get all weekly 24-hour average files, sorted by date ==========
tt_files = sorted(glob(os.path.join(input_dir, "tt_24h_*.nc")))
ss_files = sorted(glob(os.path.join(input_dir, "ss_24h_*.nc")))
ww_files = sorted(glob(os.path.join(input_dir, "ww_24h_*.nc")))

# ========== Load static coordinates once ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lat = ds1.YC.isel(face=face, i=1, j=j)
lon = ds1.XC.isel(face=face, i=i, j=1)
depth = ds1.Z
depth_kp1 = ds1.Zp1

# Broadcast to 3D
lon_b, lat_b = xr.broadcast(lon, lat)
depth3d, _, _ = xr.broadcast(depth, lon_b, lat_b)

# Horizontal grid spacing
dx = ds1.dxF.isel(face=face, i=i, j=j).mean().values
dy = ds1.dyF.isel(face=face, i=i, j=j).mean().values
dr = np.sqrt(dx**2 / 2 + dy**2 / 2)


# ========== Cross-spectrum function ==========
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



# ========== Main processing ==========
def main():
    # ========== Setup Dask distributed cluster ==========
    cluster = LocalCluster(
        n_workers=32,              # fewer workers but each with more memory
        threads_per_worker=1,
        memory_limit='10GB',        # increase per-worker memory
        local_directory="/tmp/dask",  # temp storage for spills
        dashboard_address=":8787"
    )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    # Store all daily results temporarily to disk instead of keeping in RAM
    tmp_files = []

    for week_idx, (tt_file, ss_file, ww_file) in enumerate(zip(tt_files, ss_files, ww_files)):
        date_str = os.path.basename(tt_file).split('_')[-1].replace('.nc', '')
        print(f"Processing {date_str}...")

        # load lazily
        tt = xr.open_dataset(tt_file, chunks={'time': 1, 'i': 200, 'j': 200})['Theta']
        ss = xr.open_dataset(ss_file, chunks={'time': 1, 'i': 200, 'j': 200})['Salt']
        ww = xr.open_dataset(ww_file, chunks={'time': 1, 'i': 200, 'j': 200})['W']

        # process each day sequentially (no large delayed graph)
        daily_specs = []
        for t in range(tt.sizes['time']):
            t_tt = tt.isel(time=t)
            t_ss = ss.isel(time=t)
            t_ww = ww.isel(time=t)

            spec = compute_one_spec(t_tt, t_ss, t_ww)  # compute immediately
            spec.load()  # force into memory (tiny compared to 3D field)
            daily_specs.append(spec)

            # free memory between iterations
            del t_tt, t_ss, t_ww, spec
            client.run(gc.collect)

        # concatenate and save this week’s results to disk
        week_specs = xr.concat(daily_specs, dim='time')
        week_tmp = os.path.join(output_dir, f"_tmp_week_{week_idx:02d}.nc")
        week_specs.to_netcdf(week_tmp)
        tmp_files.append(week_tmp)

        # clean up references to release memory
        del daily_specs, week_specs
        client.run(gc.collect)

    # Combine all weeks into one big dataset (on disk, not in memory)
    print("Combining all weeks...")
    all_specs = xr.open_mfdataset(tmp_files, combine='nested', concat_dim='time')

    # rolling mean (lazy)
    WB_cross_spectra_rolling = all_specs.rolling(time=7, center=True).mean()

    # save final output
    k_r = WB_cross_spectra_rolling.freq_r / dr / 1e-3
    spec_vp = WB_cross_spectra_rolling * k_r
    out_file = os.path.join(output_dir, "wb_cross_spec_vp_real_7day_rolling.nc")
    ds_out = xr.Dataset({
        'spec_vp_real': spec_vp.real,
        'k_r': k_r,
        'depth': WB_cross_spectra_rolling.Z
    })
    ds_out.to_netcdf(out_file)
    print(f"Saved final output: {out_file}")

    client.close()
    cluster.close()

# ========== Entry Point ==========
if __name__ == "__main__":
    main()

# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
from dask.distributed import Client, LocalCluster
import xrft

from set_constant import domain_name, face, i, j, start_hours, end_hours
plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
out_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/peak_wavelength_timeseries.nc"
plot_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_spectra_plots"
os.makedirs(plot_dir, exist_ok=True)

# =====================
# Reference time
# =====================
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds1 = xr.open_zarr(zarr_path, consolidated=False)
time_ref = ds1.time.isel(time=0).values  # starting datetime
total_days = (end_hours - start_hours) // 24
print(f"Total days to process: {total_days}")

# =====================
# Load all daily SSH files using open_mfdataset
# =====================
print("🔄 Loading all daily Eta files...")
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds = xr.open_mfdataset(eta_path, combine='by_coords', parallel=True)

# Estimate dx (in km) from dxC and dyC
dxC_mean = ds1.dxC.isel(face=face,i_g=i,j=j).values.mean()/1000 # Mean grid spacing in km
dyC_mean = ds1.dyC.isel(face=face,i=i,j_g=j).values.mean()/1000 # Mean grid spacing in km
dx_km = np.sqrt(0.5*(dxC_mean**2 + dyC_mean**2))

# =====================
# Loop over time steps
# =====================
dates = []
peak_wavelengths = []


for t in range(ds.dims['time']):
    eta_day = ds.Eta.isel(time=t)
    date = np.datetime_as_string(ds.time.isel(time=t).values, unit='D')
    print(f"📈 Processing {date}")

    # Remove NaNs
    eta_filled = eta_day.fillna(0.0)

    # Compute 2D power spectrum isotropic spectrum using xrft
    ps_iso = xrft.isotropic_power_spectrum(
        eta_filled,
        dim=['i', 'j'],
        spacing_tol=0.001,
        detrend='linear',
        scaling='density',
        window='hann',
        nfactor=4 
    )

    # Convert from wavenumber to wavelength (km), assuming dx ≈ dy
    k_vals = ps_iso['freq_r'].values
    with np.errstate(divide='ignore'):
        wavelengths = 1 / (k_vals * dx_km)  # wavelength in km
    wavelengths = np.nan_to_num(wavelengths, nan=np.inf)

    # Identify peak wavelength in reasonable range
    valid = (wavelengths < 500) & (wavelengths > dx_km)
    ps_vals = ps_iso.values[valid]
    peak_idx = np.argmax(ps_vals)
    peak_wavelength = wavelengths[valid][peak_idx]

    # Store
    dates.append(date)
    peak_wavelengths.append(peak_wavelength)

    # Plot spectrum
    plt.figure()
    plt.loglog(wavelengths[valid], ps_vals)
    plt.axvline(peak_wavelength, color='r', linestyle='--', label=f'Peak λ ≈ {peak_wavelength:.1f} km')
    plt.xlabel("Wavelength (km)")
    plt.ylabel("Power")
    plt.title(f"Isotropic SSH Spectrum - {date}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"xrft_spectrum_{date}.png"), dpi=200)
    plt.close()


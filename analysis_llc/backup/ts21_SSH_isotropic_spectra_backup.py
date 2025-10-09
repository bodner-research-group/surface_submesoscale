# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
# from dask.distributed import Client, LocalCluster
from scipy.fftpack import fft2, fftshift
from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# =====================
# Setup Dask cluster
# =====================
# cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
# client = Client(cluster)
# print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
out_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/peak_wavelength_timeseries.nc"
plot_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_spectra_plots"
os.makedirs(plot_dir, exist_ok=True)

# =====================
# Time axis
# =====================
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds1 = xr.open_zarr(zarr_path, consolidated=False)
time_ref = ds1.time.isel(time=0).values  # starting datetime
total_days = (end_hours - start_hours) // 24
print(f"Total days to process: {total_days}")

# =====================
# Spectral function
# =====================
def isotropic_spectrum_2d(field, dx):
    """Compute isotropic 2D spectrum using FFT"""
    ny, nx = field.shape
    field = field - np.nanmean(field)

    # Apply 2D FFT
    F = fft2(field)
    F = fftshift(F)
    psd2D = np.abs(F)**2 / (nx * ny)

    # Wavenumber grid
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dx)
    kx = fftshift(kx)
    ky = fftshift(ky)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # Bin power spectrum isotropically
    k_bins = np.linspace(0, np.max(K), min(nx, ny)//2)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    psd_iso = np.zeros_like(k_vals)

    for i in range(len(k_vals)):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        psd_iso[i] = np.nanmean(psd2D[mask])

    return k_vals, psd_iso

# Grid spacing (example for LLC4320)
dxC_mean = ds1.dxC.isel(face=face,i_g=i,j=j).values.mean()/1000 # Mean grid spacing in km
dyC_mean = ds1.dyC.isel(face=face,i=i,j_g=j).values.mean()/1000 # Mean grid spacing in km
dx = np.sqrt(0.5*(dxC_mean**2 + dyC_mean**2))

# =====================
# Loop over daily files
# =====================
dates = []
peak_wavelengths = []

for n in range(total_days):
    t0 = start_hours + n * 24
    date = np.datetime64(time_ref) + np.timedelta64(t0, 'h')
    date_str = str(date)[:10]

    eta_file = os.path.join(eta_input_dir, f"eta_24h_{date_str}.nc")
    if not os.path.exists(eta_file):
        print(f"  Missing: {eta_file}")
        continue

    print(f"Processing {date_str}")
    ds = xr.open_dataset(eta_file)
    eta = ds['Eta'].isel(time=0).values

    # Remove NaNs if present
    eta = np.nan_to_num(eta, nan=0.0)

    # Compute isotropic spectrum
    k_vals, psd_iso = isotropic_spectrum_2d(eta, dx)

    # Convert wavenumber to wavelength (km)
    wavelength_km = 1 / k_vals
    wavelength_km = np.nan_to_num(wavelength_km, nan=np.inf)

    # Find peak in the spectrum (excluding very low wavenumbers)
    valid = np.isfinite(wavelength_km) & (wavelength_km < 500)  # limit to < 500 km
    peak_idx = np.argmax(psd_iso[valid])
    peak_wavelength = wavelength_km[valid][peak_idx]

    # Store results
    dates.append(date)
    peak_wavelengths.append(peak_wavelength)

    # Plot
    plt.figure()
    plt.loglog(wavelength_km[valid], psd_iso[valid])
    plt.axvline(peak_wavelength, color='r', linestyle='--', label=f'Peak λ ≈ {peak_wavelength:.1f} km')
    plt.xlabel("Wavelength (km)")
    plt.ylabel("SSH Variance")
    plt.title(f"Isotropic SSH Spectrum - {date_str}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"spectrum_{date_str}.png"))
    plt.close()

    # Cleanup
    ds.close()

# =====================
# Save timeseries to NetCDF
# =====================
ds_out = xr.Dataset({
    "peak_wavelength": (["time"], peak_wavelengths),
    "time": (["time"], dates)
})
ds_out.to_netcdf(out_nc_path)
print(f"\n✅ Saved peak wavelength timeseries: {out_nc_path}")

# =====================
# Done
# =====================
client.close()
cluster.close()
print("🎉 All daily spectra processed.")

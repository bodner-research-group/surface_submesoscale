# ===== Imports =====
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xrft
from scipy.signal import detrend
from dask.distributed import Client, LocalCluster
from dask import delayed, compute


# from set_constant import domain_name, face, i, j, start_hours, end_hours
# ========== Time settings ==========
nday_avg = 28
delta_days = 7
start_hours = (49 + 91) * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain



plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB",
    processes=True
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
out_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/eddy_scale_timeseries-submeso_20kmCutoff.nc"
plot_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_spectra_plots"
os.makedirs(plot_dir, exist_ok=True)

# =====================
# Reference info
# =====================
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds1 = xr.open_zarr(zarr_path, consolidated=False)
time_ref = ds1.time.isel(time=0).values
total_days = (end_hours - start_hours) // 24
print(f"Total days to process: {total_days}")

# =====================
# Load all daily SSH files (lazy)
# =====================
print("🔄 Loading all daily Eta files...")
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds = xr.open_mfdataset(eta_path, combine='by_coords', parallel=True)

# =====================
# Grid spacing (km)
# =====================
dxC_mean = ds1.dxC.isel(face=face, i_g=i, j=j).values.mean() / 1000
dyC_mean = ds1.dyC.isel(face=face, i=i, j_g=j).values.mean() / 1000
dx_km = np.sqrt(0.5 * (dxC_mean**2 + dyC_mean**2))
nyquist_wavelength = 2 * dx_km
print(f"Grid spacing = {dx_km:.2f} km, Nyquist λ = {nyquist_wavelength:.2f} km")

# =====================
# Function to compute eddy scale for one time step
# =====================
@delayed
def process_one_timestep(t_index, date_str, eta_day, dx_km, plot_dir):
    """Compute isotropic power spectrum and eddy scale for one SSH snapshot."""
    try:
        # --- Fill NaNs and detrend ---
        eta_filled = eta_day.fillna(0.0).values
        eta_detrended = detrend(detrend(eta_filled, axis=0, type='linear'),
                                axis=1, type='linear')
        eta_detrended = xr.DataArray(eta_detrended, dims=('i', 'j'))

        # =====================
        # Detrend SSH field (remove large-scale background)
        # =====================
        # window_size = 181  # ~200 km cutoff, given dx ≈ 1.1 km, for mesoscales
        window_size = 20  # 20 km cutoff, given dx ≈ 1.1 km, for submesoscales
        # Remove domain mean
        eta_mean_removed = eta_day - eta_day.mean(dim=["i", "j"])
        # Remove large-scale background using rolling mean
        eta_detrended = eta_mean_removed - eta_mean_removed.rolling(
            i=window_size, j=window_size, center=True
        ).mean()
        # Fill NaNs at edges (from rolling)
        eta_detrended = eta_detrended.fillna(0.0)
        # ⚙️ rechunk before FFT
        eta_detrended = eta_detrended.chunk({'i': -1, 'j': -1})

        # --- Compute isotropic power spectrum ---
        ps_iso = xrft.isotropic_power_spectrum(
            eta_detrended,
            dim=['i', 'j'],
            detrend=None,
            scaling='density',
            window='hann',
            nfactor=1,
        )

        # --- Convert wavenumber to wavelength ---
        k_vals = ps_iso['freq_r'].values
        with np.errstate(divide='ignore', invalid='ignore'):
            wavelengths = 1 / (k_vals * dx_km)
        wavelengths = np.nan_to_num(wavelengths, nan=np.inf)

        # --- Restrict to physical range ---
        valid = (wavelengths > 2 * dx_km) & (wavelengths < 300)
        wl = wavelengths[valid]
        ps = ps_iso.values[valid]
        if len(wl) == 0 or np.sum(ps) == 0:
            return (date_str, np.nan)

        # --- Compute energy-weighted mean wavelength ---
        # 	•	If the spectrum has multiple peaks or a broad maximum, the absolute peak might be noisy or jump between adjacent bins.
        # 	•	The energy-weighted mean is smoother and more physically meaningful as a representative scale.
        # --- \lambda_{\text{eddy}} = \frac{\int P(k)\lambda(k)\,dk}{\int P(k)\,dk} ---
        # 	•	P(k) = power spectral density at wavenumber k (or equivalently wavelength λ = 1/k).
        # 	•	λ(k) = corresponding wavelength.
        # eddy_scale is basically a weighted average of all wavelengths, where each wavelength is weighted by its contribution to the total variance (energy) in the SSH field.
        #  	•	Wavelengths with more energy contribute more to the mean.
        #  	•	Wavelengths with very little energy contribute almost nothing.
        # So this gives a single number that characterizes the typical scale of SSH variability, i.e., the “eddy scale”.
        eddy_scale = np.sum(ps * wl) / np.sum(ps)   

        # --- Plot spectrum ---
        plt.figure(figsize=(6, 4))
        plt.loglog(wl, ps, label='Spectrum')
        plt.axvline(eddy_scale, color='r', linestyle='--',
                    label=f'Eddy scale ≈ {eddy_scale:.1f} km')
        plt.xlabel("Wavelength (km)")
        plt.ylabel("Power density")
        plt.title(f"Isotropic SSH Spectrum - {date_str}")
        plt.grid(True, which='both', linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.ylim(3e-11,1e-2)
        plt.savefig(os.path.join(plot_dir, f"submeso_20kmCutoff_xrft_spectrum_{date_str}.png"))
        plt.close()

        return (date_str, eddy_scale)

    except Exception as e:
        print(f"❌ Error at {date_str}: {e}")
        return (date_str, np.nan)


# =====================
# Schedule tasks
# =====================
tasks = []
for t in range(ds.dims['time']):
    date_str = np.datetime_as_string(ds.time.isel(time=t).values, unit='D')
    tasks.append(process_one_timestep(t, date_str, ds.Eta.isel(time=t), dx_km, plot_dir))

print(f"🧮 Scheduled {len(tasks)} Dask tasks...")

# =====================
# Compute in parallel
# =====================
results = compute(*tasks)

# =====================
# Save to NetCDF
# =====================
dates, eddy_scales = zip(*results)
eddy_scales = np.array(eddy_scales, dtype=float)

ds_out = xr.Dataset({
    "eddy_scale_km": (["time"], eddy_scales),
    "time": (["time"], np.array(dates, dtype='datetime64[D]'))
})
ds_out.to_netcdf(out_nc_path)
print(f"\n✅ Saved eddy scale timeseries: {out_nc_path}")

# =====================
# Cleanup
# =====================
ds.close()
client.close()
cluster.close()
print("🏁 Done!")






import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}"
output_movie = f"{figdir}/movie-SSH_submeso_20kmCutoff_xrft_spectrum.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/SSH_spectra_plots/submeso_20kmCutoff_xrft_spectrum_*.png' -vf scale=iw/2:ih/2 -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")




import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load eddy scale timeseries
# -----------------------------
nc_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/eddy_scale_timeseries-submeso_20kmCutoff.nc"
ds = xr.open_dataset(nc_path)

eddy_scale = ds['eddy_scale_km']
time = ds['time']

# -----------------------------
# Plot time series
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(time, eddy_scale, '-o', markersize=3, color='b', label='Eddy scale')
plt.xlabel('Date')
plt.ylabel('Eddy scale (km)')
plt.title('Daily Eddy Scale Time Series (remove mesoscale)')
plt.grid(True)
plt.ylim(0, np.nanmax(eddy_scale)*1.2)  # optional, leave some margin
plt.legend()
plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------
plot_file = os.path.join(figdir, f"eddy_scale_timeseries_{domain_name}.png")
plt.savefig(plot_file, dpi=200)
plt.close()

print(f"✅ Saved eddy scale time series plot: {plot_file}")




import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# ==== Constants ====
omega = 7.2921e-5  # [rad/s]

# ==== Paths ====
data_path = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/derived/strain_vorticity_daily.nc"
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/strain_vorticity"
os.makedirs(figdir, exist_ok=True)

# ==== Domain ====
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ==== Load data ====
ds = xr.open_dataset(data_path)
ds_grid = xr.open_zarr(grid_path, consolidated=False)
lat = ds_grid["YC"].sel(face=face).isel(i=i, j=j)

# ==== Compute Coriolis parameter (mean) ====
lat_rad = np.deg2rad(lat)
f0 = 2 * omega * np.sin(lat_rad)
f0_mean = f0.mean().item()
print(f"Mean f0 over domain: {f0_mean:.2e} s^-1")

# ==== Normalize variables ====
sigma_norm = ds["strain_magnitude"] / abs(f0_mean)
zeta_norm = ds["vorticity"] / f0_mean
delta_norm = ds["divergence"] / f0_mean

# ==== Daily Maps ====
print("\nGenerating daily maps...")
for t in range(len(ds.time)):
    date_str = str(ds.time[t].values)[:10]

    # Plot each field
    for field, var_norm in zip(["strain", "vorticity", "divergence"],
                               [sigma_norm, zeta_norm, delta_norm]):

        fig = plt.figure(figsize=(8, 6))
        v = var_norm.isel(time=t)
        im = plt.pcolormesh(v["i"], v["j"], v, cmap="RdBu_r", shading="auto")
        plt.title(f"{field.capitalize()} / f0 on {date_str}")
        plt.colorbar(im, label=f"{field} / f0")
        plt.xlabel("i")
        plt.ylabel("j")
        plt.tight_layout()

        outpath = os.path.join(figdir, f"{field}_norm_map_{date_str}.png")
        plt.savefig(outpath, dpi=200)
        plt.close()

    if t % 10 == 0:
        print(f"  Processed {t+1}/{len(ds.time)} days")

# ==== Weekly Joint PDFs ====
print("\nGenerating weekly joint PDFs...")
n_days = len(ds.time)
days_per_week = 7
n_weeks = n_days // days_per_week

bins = 300
sigma_range = (0, 5)
zeta_range = (-5, 5)

for w in range(n_weeks):
    t0 = w * days_per_week
    t1 = t0 + days_per_week
    week_time = ds.time[t0].values

    # Select data
    sigma_w = sigma_norm.isel(time=slice(t0, t1)).values.ravel()
    zeta_w = zeta_norm.isel(time=slice(t0, t1)).values.ravel()

    # Clean
    valid = np.isfinite(sigma_w) & np.isfinite(zeta_w)
    sigma_flat = sigma_w[valid]
    zeta_flat = zeta_w[valid]

    # Histogram
    H, xedges, yedges = np.histogram2d(sigma_flat, zeta_flat,
                                       bins=bins,
                                       range=[sigma_range, zeta_range],
                                       density=True)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    plt.pcolormesh(X, Y, H, shading='auto', cmap='viridis')
    plt.xlabel(r"$\sigma/|f_0|$")
    plt.ylabel(r"$\zeta/f_0$")
    plt.title(f"Joint PDF σ/|f₀| vs ζ/f₀ (Week {w+1}: {str(week_time)[:10]})")
    plt.colorbar(label="Joint PDF")
    plt.tight_layout()

    outname = os.path.join(figdir, f"joint_pdf_sigma_zeta_week{w+1:02d}.png")
    plt.savefig(outname, dpi=200)
    plt.close()

    if w % 5 == 0:
        print(f"  Processed week {w+1}/{n_weeks}")

print("\nAll plots saved.")

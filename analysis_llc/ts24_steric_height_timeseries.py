import os
import numpy as np
import xarray as xr
import gsw
from glob import glob
import matplotlib.pyplot as plt
from xgcm import Grid
from tqdm import tqdm
from dask.distributed import Client, LocalCluster

# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)


# ==============================================================
# Set parameters
# ==============================================================
g = 9.81
rhoConst = 1029.0
p_atm = 101325.0 / 1e4  # dbar

# Paths
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
eta_dir = os.path.join(base_dir, "surface_24h_avg")
rho_dir = os.path.join(base_dir, "rho_insitu_hydrostatic_pressure_daily")
Hml_file = os.path.join(base_dir, "Lambda_MLI_timeseries_daily.nc")

out_dir = os.path.join(base_dir, "steric_height_anomaly_timeseries")
os.makedirs(out_dir, exist_ok=True)

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/steric_height/"
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Load grid info
# ==============================================================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lon = ds1["XC"].isel(face=face, i=i, j=j)
lat = ds1["YC"].isel(face=face, i=i, j=j)
depth = ds1.Z
drF = ds1.drF
drF3d, _, _ = xr.broadcast(drF, lon, lat)

ds_grid_face = ds1.isel(face=face, i=i, j=j, i_g=i, j_g=j, k=0, k_p1=0, k_u=0)
if "time" in ds_grid_face.dims:
    ds_grid_face = ds_grid_face.isel(time=0, drop=True)

coords = {"X": {"center": "i", "left": "i_g"}, "Y": {"center": "j", "left": "j_g"}}
metrics = {("X",): ["dxC", "dxG"], ("Y",): ["dyC", "dyG"]}
grid = Grid(ds_grid_face, coords=coords, metrics=metrics, periodic=False)

# ==============================================================
# Load Eta (7-day rolling mean)
# ==============================================================
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds_eta = xr.open_mfdataset(eta_path, combine="by_coords")
Eta_daily = ds_eta["Eta"]
# Eta = Eta_daily.rolling(time=7, center=True).mean()
Eta = Eta_daily
Eta= Eta.assign_coords(time=Eta.time.dt.floor("D"))


# Load Hml time series
Hml_mean = xr.open_dataset(Hml_file).Hml_mean
Hml_mean= Hml_mean.assign_coords(time=Hml_mean.time.dt.floor("D"))

# ==============================================================
# Define helper functions
# ==============================================================
def compute_grad_laplace(var, grid):
    """Compute gradient magnitude and Laplacian using xgcm grid."""
    var_x = grid.derivative(var, axis="X")
    var_y = grid.derivative(var, axis="Y")
    var_x_c = grid.interp(var_x, axis="X", to="center")
    var_y_c = grid.interp(var_y, axis="Y", to="center")
    grad_mag = np.sqrt(var_x_c**2 + var_y_c**2)
    var_xx = grid.derivative(var_x_c, axis="X")
    var_yy = grid.derivative(var_y_c, axis="Y")
    var_xx_c = grid.interp(var_xx, axis="X", to="center")
    var_yy_c = grid.interp(var_yy, axis="Y", to="center")
    laplace = var_xx_c + var_yy_c
    return grad_mag, laplace


def find_matching_rho_file(time_val):
    """Find rho_insitu_pres_hydro_YYYYMMDD.nc matching a given Eta time."""
    date_tag = np.datetime_as_string(time_val, unit="D").replace("-", "")
    pattern = os.path.join(rho_dir, f"rho_insitu_pres_hydro_{date_tag}.nc")
    matches = glob(pattern)
    return matches[0] if matches else None


# ==============================================================
# Initialize output lists
# ==============================================================
times = []
eta_grad2_list = []
eta_prime_grad2_list = []

eta_grad_mag_all = []
eta_prime_grad_mag_all = []
eta_laplace_all = []
eta_prime_laplace_all = []

# ==============================================================
# Main loop over time steps
# ==============================================================
for t in tqdm(range(len(Eta.time)), desc="Processing time steps"):
    time_val = Eta.time.isel(time=t).values
    eta = Eta.isel(time=t)
    eta_minus_mean = eta - eta.mean(dim=["i", "j"])

    rho_file = find_matching_rho_file(time_val)
    if rho_file is None:
        print(f"⚠️ Skipping {str(time_val)} (no rho file found)")
        continue

    ds_rho = xr.open_dataset(rho_file, chunks={"k": -1, "j": 50, "i": 50})
    rho_insitu = ds_rho["rho_insitu"]
    rho_prime = rho_insitu - rho_insitu.mean(dim=("i", "j"))
    rho_prime= rho_prime.assign_coords(time=rho_prime.time.dt.floor("D"))

    # Mixed layer depth (interpolate to same time index)
    Hml_val = float(Hml_mean.sel(time=time_val, method="nearest").values)

    mask = depth >= Hml_val
    mask3d = mask.broadcast_like(rho_prime)

    rho_prime_masked = rho_prime.where(mask3d)
    drF_masked = drF3d.where(mask3d)

    # Steric height anomaly
    eta_prime = -(1 / rhoConst) * (rho_prime_masked * drF_masked).sum(dim="k")
    eta_prime_minus_mean = eta_prime - eta_prime.mean(dim=["i", "j"])

    # Compute gradients and Laplacians
    eta_grad_mag, eta_laplace = compute_grad_laplace(eta_minus_mean, grid)
    eta_prime_grad_mag, eta_prime_laplace = compute_grad_laplace(eta_prime_minus_mean, grid)

    # Store arrays for time stacking
    eta_grad_mag_all.append(eta_grad_mag)
    eta_prime_grad_mag_all.append(eta_prime_grad_mag)
    eta_laplace_all.append(eta_laplace)
    eta_prime_laplace_all.append(eta_prime_laplace)

    # Domain-mean |∇η|² and |∇η′|²
    eta_grad2_mean = (eta_grad_mag**2).mean(dim=["i", "j"])
    eta_prime_grad2_mean = (eta_prime_grad_mag**2).mean(dim=["i", "j"])
    eta_grad2_list.append(eta_grad2_mean)
    eta_prime_grad2_list.append(eta_prime_grad2_mean)
    times.append(time_val)

# ==============================================================
# Stack over time
# ==============================================================

eta_grad_mag_all = xr.concat(eta_grad_mag_all, dim="time")
eta_prime_grad_mag_all = xr.concat(eta_prime_grad_mag_all, dim="time")
eta_laplace_all = xr.concat(eta_laplace_all, dim="time")
eta_prime_laplace_all = xr.concat(eta_prime_laplace_all, dim="time")

ds_out = xr.Dataset(
    {
        "eta": eta,
        "eta_grad_mag": eta_grad_mag_all,
        "eta_laplace": eta_laplace_all,
        "eta_prime": eta_prime,
        "eta_prime_grad_mag": eta_prime_grad_mag_all,
        "eta_prime_laplace": eta_prime_laplace_all,
    }
)
ds_out["lon"] = lon
ds_out["lat"] = lat
ds_out["time"] = ("time", times)
ds_out.to_netcdf(os.path.join(out_dir, "grad_laplace_eta_steric_daily.nc"), compute=True)
print("✅ Saved spatial gradients and Laplacians: grad_laplace_eta_steric_daily.nc")

# ==============================================================
# Domain-mean time series
# ==============================================================
ts_ds = xr.Dataset(
    {
        "eta_grad2_mean": xr.concat(eta_grad2_list, dim="time"),
        "eta_prime_grad2_mean": xr.concat(eta_prime_grad2_list, dim="time"),
    },
    coords={"time": ("time", times)},
)
ts_ds.to_netcdf(os.path.join(out_dir, "grad2_timeseries.nc"))
print("✅ Saved domain-mean |∇η|² and |∇η′|² timeseries: grad2_timeseries.nc")

# ==============================================================
# Plot timeseries
# ==============================================================

plt.figure(figsize=(8, 4))
ts_ds["eta_grad2_mean"].plot(label="⟨|∇η|²⟩", color="tab:blue")
ts_ds["eta_prime_grad2_mean"].plot(label="⟨|∇η′|²⟩", color="tab:orange")

# 7-day rolling mean
ts_ds["eta_grad2_mean"].rolling(time=7, center=True).mean().plot(label="⟨|∇η|²⟩ (7d)", color="tab:blue", linestyle="--")
ts_ds["eta_prime_grad2_mean"].rolling(time=7, center=True).mean().plot(label="⟨|∇η′|²⟩ (7d)", color="tab:orange", linestyle="--")

plt.title("Domain-mean |∇η|² and |∇η′|² Timeseries")
plt.ylabel("Mean(|∇η|²) [m²/m²]")
plt.xlabel("Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{figdir}grad2_timeseries.png", dpi=150)
plt.close()
print(f"✅ Saved figure: {figdir}grad2_timeseries.png")

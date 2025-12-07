
import os
import numpy as np
import xarray as xr
import gsw
from glob import glob
import matplotlib.pyplot as plt
from xgcm import Grid
from tqdm import tqdm

from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ==============================================================
# Set parameters
# ==============================================================
g = 9.81
rho0 = 1027.5
p_atm = 101325.0 / 1e4  # dbar

# Paths
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
eta_dir = os.path.join(base_dir, "surface_24h_avg")
# rho_dir = os.path.join(base_dir, "rho_insitu_hydrostatic_pressure_daily")
# Hml_file = os.path.join(base_dir, "Lambda_MLI_timeseries_daily.nc")

out_dir = os.path.join(base_dir, "SSH_submesoscale")
os.makedirs(out_dir, exist_ok=True)

figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_submesoscale/"
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

shortname = "SSH_Gaussian_submeso_14kmCutoff"
fname = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/{shortname}.nc" 
eta_submeso = xr.open_dataset(fname).SSH_submesoscale
# eta_submeso = xr.open_dataset(fname).eta_submeso ### Only for GCMFilters
# eta_submeso = xr.open_dataset(fname).SSH_mesoscale
# eta_submeso = xr.open_dataset(fname).eta_meso ### Only for GCMFilters
eta_submeso= eta_submeso.assign_coords(time=eta_submeso.time.dt.floor("D"))

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


# ==============================================================
# Initialize output lists
# ==============================================================
times = []
eta_submeso_grad2_list = []

# ==============================================================
# Main loop over time steps
# ==============================================================
for t in tqdm(range(len(Eta.time)), desc="Processing time steps"):
    time_val = Eta.time.isel(time=t).values
    date_tag = np.datetime_as_string(time_val, unit="D").replace("-", "")

    # day_file = os.path.join(out_dir, f"grad_laplace_eta_submeso_{date_tag}.nc")
    # if os.path.exists(day_file):
    #     print(f"Already processed: {day_file}")
    #     continue

    # eta = Eta.isel(time=t)
    # eta_minus_mean = eta - eta.mean(dim=["i", "j"])

    eta_submeso_t = eta_submeso.isel(time=t)

    # Compute gradients and Laplacians
    eta_submeso_grad_mag, eta_submeso_laplace = compute_grad_laplace(eta_submeso_t, grid)

    # Domain-mean submesoscale |∇η′|². Exclude 2 grid points on each boundary (mask edges)
    eta_submeso_grad2_mean = (eta_submeso_grad_mag.isel(i=slice(2, -2), j=slice(2, -2))**2).mean(dim=["i", "j"])
    eta_submeso_grad2_list.append(eta_submeso_grad2_mean)
    times.append(time_val)

    # # ==============================================================
    # # Save per-day output file
    # # ==============================================================
    # ds_day = xr.Dataset(
    #     {
    #         "eta_submeso": eta_submeso,
    #         "eta_submeso_grad_mag": eta_submeso_grad_mag,
    #         "eta_submeso_laplace": eta_submeso_laplace,
    #     },
    #     coords={
    #         "lon": lon,
    #         "lat": lat,
    #         "time": ("time", [time_val]),
    #     },
    # )

    # ds_day.to_netcdf(day_file)
    # print(f"✅ Saved {day_file}")


# ==============================================================
# Domain-mean time series
# ==============================================================
ts_ds = xr.Dataset(
    {
        "eta_submeso_grad2_mean": xr.concat(eta_submeso_grad2_list, dim="time"),
    },
    coords={"time": ("time", times)},
)
ts_ds.to_netcdf(os.path.join(out_dir, f"{shortname}_timeseries.nc"))
print(f"✅ Saved domain-mean |∇η_submeso|² timeseries: {shortname}_timeseries.nc")

# ==============================================================
# Plot timeseries
# ==============================================================

# ts_ds = ts_ds.chunk({"time": 365})  # or any chunk >= 7

plt.figure(figsize=(8, 4))
ts_ds["eta_submeso_grad2_mean"].plot(label="⟨submesoscale |∇η′|²⟩", color="tab:orange")

# 7-day rolling mean
ts_ds["eta_submeso_grad2_mean"].compute().rolling(time=7, center=True).mean().plot(label="⟨submesoscale |∇η′|²⟩ (7d)", color="tab:orange", linestyle="--")

plt.title("Domain-mean submesoscale |∇η′|² Timeseries")
plt.ylabel("Mean(submesoscale |∇η′|²) [m²/m²]")
plt.xlabel("Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{figdir}{shortname}_timeseries.png", dpi=150)
plt.close()
print(f"✅ Saved figure: {figdir}{shortname}_timeseries.png")












# #### Compute and save the domain-mean ⟨|∇η′|²⟩ time series (using saved eta_submeso_grad_mag)
# #### following Wang et al. 2025

# import os
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# from glob import glob
# from tqdm import tqdm

# from set_constant import domain_name, face, i, j
# # # ========== Domain ==========
# # domain_name = "icelandic_basin"
# # face = 2
# # i = slice(527, 1007)   # icelandic_basin -- larger domain
# # j = slice(2960, 3441)  # icelandic_basin -- larger domain

# # ==============================================================
# # Paths and constants
# # ==============================================================
# base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
# out_dir = os.path.join(base_dir, "steric_height_anomaly_timeseries")
# os.makedirs(out_dir, exist_ok=True)

# figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/steric_height/"
# os.makedirs(figdir, exist_ok=True)

# # ==============================================================
# # Find all daily files containing eta_submeso_grad_mag
# # ==============================================================
# file_list = sorted(glob(os.path.join(out_dir, "grad_laplace_eta_submeso_*.nc")))
# if len(file_list) == 0:
#     raise FileNotFoundError(f"No grad_laplace_eta_submeso_*.nc files found in {out_dir}")

# print(f"Found {len(file_list)} daily files to process.")

# # ==============================================================
# # Initialize output lists
# # ==============================================================
# times = []
# eta_submeso_grad2_mean_list = []

# # ==============================================================
# # Main loop: read, mask boundaries, compute mean(|∇η′|²)
# # ==============================================================
# for f in tqdm(file_list, desc="Computing mean |∇η′|²"):
#     ds = xr.open_dataset(f)
#     time_val = ds.time.values[0] if "time" in ds else np.nan

#     # Read eta_submeso_grad_mag
#     grad_mag = ds["eta_submeso_grad_mag"]

#     # Exclude 2 grid points on each boundary (mask edges)
#     grad_mag_center = grad_mag.isel(i=slice(2, -2), j=slice(2, -2))

#     # Compute domain-mean of |∇η′|²
#     grad2_mean = (grad_mag_center ** 2).mean(dim=["i", "j"])
#     eta_submeso_grad2_mean_list.append(grad2_mean)
#     times.append(time_val)

#     ds.close()

# # ==============================================================
# # Combine results into one dataset
# # ==============================================================
# ts_ds = xr.Dataset(
#     {"eta_submeso_grad2_mean": xr.concat(eta_submeso_grad2_mean_list, dim="time")},
#     coords={"time": ("time", times)},
# )
# out_ts_file = os.path.join(out_dir, "grad2_submeso_timeseries.nc")
# ts_ds.to_netcdf(out_ts_file)
# print(f"✅ Saved domain-mean |∇η′|² timeseries: {out_ts_file}")

# # ==============================================================
# # Plot timeseries
# # ==============================================================
# plt.figure(figsize=(8, 4))
# ts_ds["eta_submeso_grad2_mean"].plot(label="⟨submesoscale |∇η′|²⟩", color="tab:orange")

# # 7-day rolling mean
# ts_ds["eta_submeso_grad2_mean"].rolling(time=7, center=True).mean().plot(
#     label="⟨submesoscale |∇η′|²⟩ (7d)", color="tab:orange", linestyle="--"
# )

# plt.title("Domain-mean submesoscale |∇η′|² Timeseries (Boundary-excluded)")
# plt.ylabel("Mean(submesoscale |∇η′|²) [m²/m²]")
# plt.xlabel("Time")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# fig_file = os.path.join(figdir, "grad2_submeso_timeseries.png")
# plt.savefig(fig_file, dpi=150)
# plt.close()
# print(f"✅ Saved figure: {fig_file}")

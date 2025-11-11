### Cite GCM-Filters:
### Loose et al., (2022). GCM-Filters: A Python Package for Diffusion-based Spatial Filtering of Gridded Data. Journal of Open Source Software, 7(70), 3947, https://doi.org/10.21105/joss.03947
### Grooms et al., (2021). Diffusion-Based Smoothers for Spatial Filtering of Gridded Geophysical Data. Journal of Advances in Modeling Earth Systems, 13, e2021MS002552, https://doi.org/10.1029/2021MS002552

# ===== Imports =====
import os
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask import delayed, compute
import gcm_filters as gf   # the spatial filtering package

# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ========== Time settings ==========
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(
    n_workers=32,
    threads_per_worker=1,
    memory_limit="11GB",
    processes=True
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
out_nc_path_meso = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_GCMFilters_meso_30kmCutoff.nc"
out_nc_path_submeso = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_GCMFilters_submesoscale_30kmCutoff.nc"
out_nc_path_meso_coarse = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/SSH_GCMFilters_meso_30kmCutoff_1_12deg.nc"

plot_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_submesoscale"
os.makedirs(plot_dir, exist_ok=True)

# =====================
# Reference info
# =====================
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds1 = xr.open_zarr(zarr_path, consolidated=False)
time_ref = ds1.time.isel(time=0).values
print("Opened reference dataset.")
# compute grid spacing (km)
dxC_mean = ds1.dxC.isel(face=face, i_g=i, j=j).values.mean() / 1000
dyC_mean = ds1.dyC.isel(face=face, i=i, j_g=j).values.mean() / 1000
dx_km = np.sqrt(0.5 * (dxC_mean**2 + dyC_mean**2))
print(f"Grid spacing ~ {dx_km:.2f} km")

# =====================
# Load all daily SSH files (lazy)
print("🔄 Loading all daily Eta files...")
eta_path = os.path.join(eta_dir, "eta_24h_*.nc")
ds = xr.open_mfdataset(eta_path, combine='by_coords', parallel=True)
ssh = ds.Eta
print("Data loaded.")

# =====================
# Function: filter, save meso & submeso, coarse‐grain meso
@delayed
def process_one_timestep(t_index, date_str, eta_day, dx_km):
    """
    Use gcm-filters to compute ‘mesoscale’ field (eta_meso) and derive submesoscale:
    eta_meso = large‐scale via filter (cutoff ~ 30 km)
    eta_submeso = eta_day minus eta_meso
    Then coarse‐grain eta_meso (4×4 → 1/12°) and return all three.
    """
    try:
        # Remove domain mean
        eta_mean_removed = eta_day - eta_day.mean(dim=["i", "j"])

        # Define cutoff scale
        cutoff_km = 30.0

        # ---- Construct GCM Filter object ----
        # According to docs, we must define grid spacings in meters.
        dx_m = dx_km * 1000.0
        dy_m = dx_km * 1000.0  # assuming roughly isotropic

        # The smoothing length scale (in meters)
        filter_scale_m = cutoff_km * 1000.0

        # Create filter instance:
        filter_obj = gf.Filter(
            filter_scale=filter_scale_m,
            dx_min=dx_m,
            filter_shape=gf.FilterShape.GAUSSIAN,   # Gaussian smoother
            grid_type=gf.GridType.REGULAR,          # regular lat/lon or cartesian grid
        )

        # ---- Apply the filter ----
        eta_meso = filter_obj.apply(eta_mean_removed, dims=("i", "j"))
        eta_meso = eta_meso.rename("eta_meso")
        eta_meso.attrs["description"] = f"SSH mesoscale (> {cutoff_km} km)"
        eta_meso.attrs["filter_cutoff_km"] = cutoff_km

        # ---- Submesoscale residual ----
        eta_submeso = (eta_mean_removed - eta_meso).rename("eta_submeso")
        eta_submeso.attrs["description"] = f"SSH submesoscale (< {cutoff_km} km)"

        # ---- Coarse-grain mesoscale field ----
        coarse_factor = 4
        eta_meso_coarse = (
            eta_meso.coarsen(i=coarse_factor, j=coarse_factor, boundary="trim")
            .mean()
            .rename("eta_meso_coarse")
        )
        eta_meso_coarse.attrs["description"] = f"Mesoscale SSH coarse-grained (~1/12°)"

        return date_str, eta_meso, eta_submeso, eta_meso_coarse

    except Exception as e:
        print(f"⚠️ Error processing {date_str}: {e}")
        nan_meso = xr.full_like(eta_day, np.nan).rename("eta_meso")
        nan_sub = xr.full_like(eta_day, np.nan).rename("eta_submeso")
        nan_coarse = xr.full_like(
            eta_day.coarsen(i=4, j=4, boundary="trim").mean(), np.nan
        ).rename("eta_meso_coarse")
        return date_str, nan_meso, nan_sub, nan_coarse

# =====================
# Schedule tasks
tasks = []
for t in range(ds.dims["time"]):
    date_str = np.datetime_as_string(ds.time.isel(time=t).values, unit="D")
    eta_day = ssh.isel(time=t)
    tasks.append(process_one_timestep(t, date_str, eta_day, dx_km))

print(f"🧮 Scheduled {len(tasks)} tasks")

# =====================
# Compute in parallel
results = compute(*tasks)
dates, meso_list, submeso_list, meso_coarse_list = zip(*results)

# =====================
# Concatenate and save
SSH_meso = xr.concat(meso_list, dim="time")
SSH_meso = SSH_meso.assign_coords(time=("time", np.array(dates, dtype="datetime64[D]")))

SSH_submeso = xr.concat(submeso_list, dim="time")
SSH_submeso = SSH_submeso.assign_coords(time=("time", np.array(dates, dtype="datetime64[D]")))

SSH_meso_coarse = xr.concat(meso_coarse_list, dim="time")
SSH_meso_coarse = SSH_meso_coarse.assign_coords(time=("time", np.array(dates, dtype="datetime64[D]")))

# Create datasets
ds_out_meso = xr.Dataset({"eta_meso": SSH_meso})
ds_out_sub = xr.Dataset({"eta_submeso": SSH_submeso})
ds_out_meso_coarse = xr.Dataset({"eta_meso_coarse": SSH_meso_coarse})

# Save to netCDF
ds_out_meso.to_netcdf(out_nc_path_meso)
print(f"✅ Saved mesoscale SSH dataset: {out_nc_path_meso}")

ds_out_sub.to_netcdf(out_nc_path_submeso)
print(f"✅ Saved submesoscale SSH dataset: {out_nc_path_submeso}")

ds_out_meso_coarse.to_netcdf(out_nc_path_meso_coarse)
print(f"✅ Saved coarse‐grained mesoscale SSH dataset: {out_nc_path_meso_coarse}")

# =====================
# Cleanup
ds.close()
ds1.close()
client.close()
cluster.close()
print("🏁 Done!")

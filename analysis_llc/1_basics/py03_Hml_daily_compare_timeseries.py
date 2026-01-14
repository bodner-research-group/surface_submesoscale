import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from glob import glob


# =========================================================
# User options
# =========================================================
from set_constant import domain_name, face, i, j

data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
out_dir  = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
fig_dir  = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Hml_daily_compare"
os.makedirs(fig_dir, exist_ok=True)


# Global font size setting for figures
plt.rcParams.update({'font.size': 16})


# =========================================================
# Main
# =========================================================
# def main():

# Load single grid slice for lat/lon
ds_grid = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)

# File lists
# Surface reference (new)
files_surface = sorted(glob(os.path.join(data_dir, "Hml_daily_surface_reference_*.nc")))
# 10m reference (original)
files_10m = sorted(glob(os.path.join(data_dir, "rho_Hml_TS_daily_*.nc")))

print(f"Found {len(files_surface)} surface-ref Hml files")
print(f"Found {len(files_10m)} 10m-ref Hml files")

times = []
Hml_surface = []
Hml_10m = []

for f_surface in files_surface:

    date_tag = os.path.basename(f_surface).split("_")[-1].replace(".nc", "")

    f_10m = os.path.join(data_dir, f"rho_Hml_TS_daily_{date_tag}.nc")
    if not os.path.exists(f_10m):
        print("Missing 10m reference file for", date_tag)
        continue

    # Load surface reference Hml
    ds_s = xr.open_dataset(f_surface)
    Hml_surface.append(np.abs(ds_s["Hml_daily"].values))   # (j,i)

    # Load 10m reference Hml
    ds_10 = xr.open_dataset(f_10m)
    Hml_10m.append(np.abs(ds_10["Hml_daily"].values))      # (j,i)

    times.append(np.datetime64(f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:]}", 'D'))

    ds_s.close()
    ds_10.close()



# =========================================================
# Convert lists → numpy arrays → xarray DataArrays
# =========================================================
Hml_surface_arr = np.stack(Hml_surface, axis=0)   # shape (time, j, i)
Hml_10m_arr     = np.stack(Hml_10m, axis=0)

times64 = np.array(times, dtype='datetime64[s]')

Hml_surface_da = xr.DataArray(
    Hml_surface_arr,
    dims=("time", "j", "i"),
    coords={"time": times64}
)

Hml_10m_da = xr.DataArray(
    Hml_10m_arr,
    dims=("time", "j", "i"),
    coords={"time": times64}
)


# =========================================================
# Horizontal means
# =========================================================
Hsurf_mean = Hml_surface_da.mean(dim=("j", "i"), skipna=True)
H10m_mean  = Hml_10m_da.mean(dim=("j", "i"), skipna=True)


# Find surface Hml values < 1
mask_surf = Hsurf_mean < 1
if mask_surf.any():
    values_surf = Hsurf_mean.where(mask_surf, drop=True).values  # values < 1
    print(f"Warning: surface Hml values < 1 on: {values_surf}")
# Find 10m Hml values < 1
mask_10m = H10m_mean < 1
if mask_10m.any():
    values_10m = H10m_mean.where(mask_10m, drop=True).values
    print(f"Warning: 10m Hml values < 1 on: {values_10m}")

indices = np.argwhere(mask_surf.values)
print(f"indices of surface Hml < 1: {indices}")

indices = np.argwhere(mask_10m.values)
print(f"indices of 10m Hml < 1: {indices}")

# =========================================================
# Time derivatives (xarray handles dt correctly)
# =========================================================
dHsurf_dt = Hsurf_mean.differentiate("time") * 86400.0  # m/s → m/day
dH10m_dt  = H10m_mean.differentiate("time") * 86400.0


# =========================================================
# Build dataset for output
# =========================================================
ds_out = xr.Dataset(
    {
        "Hml_surface_mean" : Hsurf_mean,
        "Hml_10m_mean"     : H10m_mean,
        "dHml_surface_dt"  : dHsurf_dt,
        "dHml_10m_dt"      : dH10m_dt,
    }
)

# =========================================================
# Save
# =========================================================
nc_path = os.path.join(out_dir, "Hml_timeseries.nc")
ds_out.to_netcdf(nc_path)
print("Saved mean time series NetCDF:", nc_path)



# =========================================================
# Plot figure
# =========================================================
fig, ax = plt.subplots(2,1, figsize=(14,10), constrained_layout=True)

ax[0].plot(ds_out.time, ds_out.Hml_surface_mean, label="Surface reference", color="tab:blue")
ax[0].plot(ds_out.time, ds_out.Hml_10m_mean,     label="10m reference", color="tab:red")
ax[0].set_title("Mixed Layer Depth (Horizontal Mean)")
ax[0].set_ylabel("Hml (m)")
ax[0].legend(); ax[0].grid()

ax[1].plot(ds_out.time, ds_out.dHml_surface_dt, label="Surface reference", color="tab:blue")
ax[1].plot(ds_out.time, ds_out.dHml_10m_dt,     label="10m reference", color="tab:red")
ax[1].set_title("dHml/dt (Horizontal Mean, m/day)")
ax[1].set_ylabel("m/day"); ax[1].set_xlabel("Time")
ax[1].legend(); ax[1].grid()

fig_path = os.path.join(fig_dir, "Hml_timeseries.png")
fig.savefig(fig_path, dpi=150)
plt.close()
print("Saved figure:", fig_path)


# # =========================================================
# # Entry point
# # =========================================================
# if __name__ == "__main__":
#     main()

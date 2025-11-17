##### Compute the contribution of Ekman buoyancy flux to the change in mixed layer depth
##### Following Thomas (2005), Thomas & Lee (2005), Thomas & Ferrari (2008), Thompson et al. (2016), Johnson et al. (2020b), etc.
#####
##### The change in mixed layer stratification (\partial_t b_z)^{Ek} ~ (\tau^x\partial_y b - \tau^y\partial_x b)/rho0/f/h^2, where h is a depth. 
##### h can be the mixed layer depth, the convective layer depth, the Ekman layer depth, or the KPP boundary layer depth??
#####

import os
import numpy as np
import xarray as xr
from glob import glob
from xgcm import Grid

# ========== Domain ==========
# from set_constant import domain_name, face, i, j
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ==============================================================
# Constants
# ==============================================================
g = 9.81
rho0 = 1027.5

# ==============================================================
# Paths
# ==============================================================
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
rho_dir  = os.path.join(base_dir, "rho_insitu_hydrostatic_pressure_daily")
hml_file = os.path.join(base_dir, "Lambda_MLI_timeseries_daily.nc")
out_dir  = os.path.join(base_dir, "ekman_buoyancy_flux_daily")
os.makedirs(out_dir, exist_ok=True)

# ==============================================================
# Load grid + wind stress (Dask lazy)
# ==============================================================
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

lon  = ds1.XC.isel(face=face, i=i, j=j)
lat  = ds1.YC.isel(face=face, i=i, j=j)
depth = ds1.Z
drF   = ds1.drF
drF3d, _, _ = xr.broadcast(drF, lon, lat)

Coriolis = 4*np.pi/86164*np.sin(lat*np.pi/180)
f0 = Coriolis.mean(dim=("i","j")).values     ### Coriolis parameter averaged over this domain

# Wind (chunk over time)
taux = ds1.oceTAUX.isel(face=face, j=j, i_g=i).chunk({"time": 24})
tauy = ds1.oceTAUY.isel(face=face, j_g=j, i=i).chunk({"time": 24})

# xgcm grid
ds_grid = ds1.isel(face=face, i=i, j=j, i_g=i, j_g=j, k=0, k_u=0, k_p1=0)
if "time" in ds_grid.dims:
    ds_grid = ds_grid.isel(time=0, drop=True)

coords = {"X": {"center": "i", "left": "i_g"},
          "Y": {"center": "j", "left": "j_g"}}
metrics = {("X",): ["dxC", "dxG"], ("Y",): ["dyC", "dyG"]}
grid = Grid(ds_grid, coords=coords, metrics=metrics, periodic=False)

# ==============================================================
# Helpers
# ==============================================================

def grad_center(var):
    dx = grid.derivative(var, axis="X")
    dy = grid.derivative(var, axis="Y")
    dx = grid.interp(dx, axis="X", to="center")
    dy = grid.interp(dy, axis="Y", to="center")
    return dx, dy

def rho_file(date):
    tag = np.datetime_as_string(date, unit="D").replace("-", "")
    f = glob(os.path.join(rho_dir, f"rho_insitu_pres_hydro_{tag}.nc"))
    return f[0] if f else None

# Mixed layer depth daily
Hml_daily = xr.open_dataset(hml_file).Hml_mean
Hml_daily = Hml_daily.assign_coords(time=Hml_daily.time.dt.floor("D"))

# ==============================================================
# DAILY PROCESSING LOOP (low overhead, Dask does real work)
# ==============================================================

days = np.unique(taux.time.dt.floor("D").values)

for day in days:
    date = np.datetime_as_string(day, unit="D").replace("-", "")
    outfile = os.path.join(out_dir, f"B_Ekman_{date}.nc")
    if os.path.exists(outfile):
        continue

    # Select wind stress for that day (may be 24 hr or more)
    mask = taux.time.dt.floor("D") == day
    taux_d = taux.sel(time=mask)
    tauy_d = tauy.sel(time=mask)

    # Find density file
    rf = rho_file(day)
    if rf is None:
        print(f"⚠ No density file found for {date}")
        continue

    # Load daily density
    ds_rho = xr.open_dataset(rf).chunk({"k": -1, "j": 50, "i": 50})
    rho = ds_rho.rho_insitu.isel(i=i, j=j)

    # Density anomaly
    rho_prime = rho - rho.mean(("i", "j"))

    # Mixed layer depth
    Hml = float(Hml_daily.sel(time=day, method="nearest").values)

    # ML mask
    ML = depth >= Hml
    ML3 = ML.broadcast_like(rho_prime)
    drF_ML = drF3d.where(ML3)

    # ML-averaged buoyancy anomaly
    rho_ML = (rho_prime.where(ML3) * drF_ML).sum("k") / drF_ML.sum("k")
    b = -g * rho_ML / rho0   # (i,j)

    # Gradients (lazy)
    bx, by = grad_center(b)

    # Wind stress -> center
    taux_c = grid.interp(taux_d, axis="X", to="center")
    tauy_c = grid.interp(tauy_d, axis="Y", to="center")

    # Ekman buoyancy flux (lazy Dask)
    B = (taux_c * by - tauy_c * bx) / (rho0 * Coriolis)   # (time,i,j)

    # Daily mean (lazy)
    B_daily = B.mean("time").compute()  # compute only final product

    # Save
    ds_out = xr.Dataset(
        {"B_Ek": B_daily},
        coords={"lon": lon, "lat": lat, "time": [day]},
    )
    ds_out.to_netcdf(outfile)
    print(f"✔ Saved {outfile}")

print("✓ All daily Ekman buoyancy flux files computed.")

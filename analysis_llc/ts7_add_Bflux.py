import xarray as xr
import gsw
from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# === Paths ===
input_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day.nc"
output_nc_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/qnet_fwflx_daily_7day_Bflux.nc" 

# === Load Dataset ===
ds = xr.open_dataset(input_nc_path)

# === Constants ===
gravity = 9.81       # m/s^2
SA = 35.2            # Absolute Salinity [g/kg]
CT = 6               # Conservative Temperature [°C]
p = 0                # Sea pressure [dbar]
rho0 = 999.8         # Reference density [kg/m^3]
Cp = 3975            # Heat capacity [J/kg/K]

# === Equation of state parameters ===
beta = gsw.beta(SA, CT, p)
alpha = gsw.alpha(SA, CT, p)

# === Buoyancy Flux Computation ===
Bflux_daily_avg = gravity * alpha * ds.qnet_daily_avg / (rho0 * Cp) + \
                  gravity * beta * ds.fwflx_daily_avg * SA / rho0

Bflux_7day_smooth = gravity * alpha * ds.qnet_7day_smooth / (rho0 * Cp) + \
                    gravity * beta * ds.fwflx_7day_smooth * SA / rho0

# === Assign attributes ===
Bflux_daily_avg.name = "Bflux_daily_avg"
Bflux_daily_avg.attrs["units"] = "m^2/s^3"
Bflux_daily_avg.attrs["long_name"] = "Daily average surface buoyancy flux"

Bflux_7day_smooth.name = "Bflux_7day_smooth"
Bflux_7day_smooth.attrs["units"] = "m^2/s^3"
Bflux_7day_smooth.attrs["long_name"] = "7-day smoothed surface buoyancy flux"

# === Add to dataset ===
ds["Bflux_daily_avg"] = Bflux_daily_avg
ds["Bflux_7day_smooth"] = Bflux_7day_smooth

# === Save updated dataset ===
ds.to_netcdf(output_nc_path)
print(f"Updated NetCDF with Bflux fields saved to: {output_nc_path}")

import os
import xarray as xr
from glob import glob
from tqdm import tqdm

# ==============================================================
# Domain
# ==============================================================
domain_name = "icelandic_basin"

# ==============================================================
# Paths
# ==============================================================
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
in_dir = os.path.join(base_dir, "steric_height_anomaly_timeseries_surface_reference")

outdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory"
os.makedirs(outdir, exist_ok=True)
out_file = os.path.join(outdir, "eta_steric_grad_mag_daily.nc")


# ==============================================================
# Find daily files
# ==============================================================
file_list = sorted(
    glob(os.path.join(in_dir, "grad_laplace_eta_steric_*.nc"))
)

if len(file_list) == 0:
    raise FileNotFoundError("No daily files found")

print(f"✅ Found {len(file_list)} daily files")

# ==============================================================
# Containers
# ==============================================================
eta_prime_list = []
times = []

# ==============================================================
# Loop over files
# ==============================================================
for f in tqdm(file_list, desc="Collecting eta_prime_grad_mag"):
    ds = xr.open_dataset(f)

    if "time" not in ds.coords:
        raise ValueError(f"No time coordinate in {f}")

    # extract time
    time_val = ds.time.values[0]

    # extract variable
    eta_prime = ds["eta_prime_grad_mag"]

    eta_prime_list.append(eta_prime)
    times.append(time_val)

    ds.close()

# ==============================================================
# Combine and save
# ==============================================================
eta_prime_ds = xr.concat(eta_prime_list, dim="time")
eta_prime_ds = eta_prime_ds.assign_coords(time=("time", times))

eta_prime_ds.to_netcdf(out_file)

print(f"✅ Saved eta_prime_grad_mag to: {out_file}")

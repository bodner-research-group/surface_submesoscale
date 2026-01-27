import os
import xarray as xr
import pandas as pd
from glob import glob

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ===================
# Paths
# ===================
in_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_7d_rolling"

outdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/VHF_theory"
os.makedirs(outdir, exist_ok=True)

out_file = os.path.join(outdir, "TuH_deg_7d_rolling_all.nc")

# ===================
# Find files
# ===================
files = sorted(glob(os.path.join(in_dir, "TuVH_7d_*.nc")))

if len(files) == 0:
    raise FileNotFoundError("No TuVH_7d_*.nc files found")

TuH_list = []
time_list = []

# ===================
# Load TuH_deg only
# ===================
for f in files:
    date_tag = os.path.basename(f).split("_")[-1].replace(".nc", "")
    time_list.append(pd.to_datetime(date_tag))

    ds = xr.open_dataset(f)
    TuH_list.append(ds["TuH_deg"])
    ds.close()

# ===================
# Concatenate along time
# ===================
TuH_all = xr.concat(TuH_list, dim="time")
TuH_all = TuH_all.assign_coords(time=("time", time_list))

# ===================
# Save
# ===================
ds_out = xr.Dataset(
    {
        "TuH_deg": TuH_all
    }
)

ds_out.to_netcdf(out_file)
print(f"Saved combined TuH file: {out_file}")

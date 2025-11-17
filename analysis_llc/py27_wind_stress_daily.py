##### Compute daily averaged surface wind stress into ONE NetCDF file

import xarray as xr
import numpy as np
import os
import time
from dask.distributed import Client, LocalCluster

from set_constant import start_hours, end_hours

# ========== Domain settings ==========
# from set_constant import domain_name, face, i, j
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)


# ========== Paths ==========
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
os.makedirs(output_dir, exist_ok=True)
outfile = os.path.join(output_dir, "surface_windstress_24h.nc")

# ========== Open dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# ========== Dask cluster ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Helper: extract a single variable ==========
def extract_and_average(varname, i_name, j_name):
    print(f"\n Selecting and averaging {varname}")

    da = ds1[varname].isel(face=face, **{i_name: i, j_name: j})

    if 'k' in da.dims:      # only for oceTAUX/TAUY (surface)
        da = da.isel(k=0)

    # full time window
    da = da.isel(time=slice(start_hours, end_hours))

    # coarsen to daily (24h) average
    da_24h = da.coarsen(time=24, boundary='trim').mean()

    return da_24h

# ========== Main ==========
if __name__ == "__main__":

    # Extract 24h averaged taux and tauy
    taux_24h = extract_and_average("oceTAUX", "i_g", "j")
    tauy_24h = extract_and_average("oceTAUY", "i", "j_g")

    # Combine into dataset
    ds_out = xr.Dataset(
        {
            "taux": taux_24h,
            "tauy": tauy_24h,
        }
    )

    print("\n Computing and writing output NetCDF...")
    t0 = time.time()

    # Write a SINGLE file
    ds_out.compute().to_netcdf(outfile)

    print(f"\n DONE. File saved:\n   {outfile}")
    print(f"Total time: {(time.time() - t0)/60:.2f} minutes")

    client.close()
    cluster.close()

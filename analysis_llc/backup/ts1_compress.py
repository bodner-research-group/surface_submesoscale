#########################
######################### Compress the data
#########################
import xarray as xr
import os
from glob import glob
from dask.distributed import Client, LocalCluster
import dask

# ========== Dask cluster setup ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

input_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin"
output_dir = os.path.join(input_dir, "compressed")
os.makedirs(output_dir, exist_ok=True)

file_patterns = ["tt_12h_*.nc", "ss_12h_*.nc", "ww_12h_*.nc"]

comp = dict(zlib=True, complevel=4)  

def compress_file(fpath):
    print(f"Start compressing {fpath}")
    ds = xr.open_dataset(fpath, chunks={})
    encoding = {var: comp for var in ds.data_vars}
    fname = os.path.basename(fpath)
    outpath = os.path.join(output_dir, fname)
    ds.to_netcdf(outpath, encoding=encoding, engine='netcdf4')
    ds.close()
    print(f"Finished compressing {fpath}")
    return outpath

# all path
all_files = []
for pattern in file_patterns:
    all_files.extend(sorted(glob(os.path.join(input_dir, pattern))))

# use dask.delayed 
tasks = [dask.delayed(compress_file)(f) for f in all_files]

dask.compute(*tasks)

client.close()
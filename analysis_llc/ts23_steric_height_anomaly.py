#### Compute the steric height anomaly, following ECCO V4 Python Tutorial
#### Andrew Delman: https://ecco-v4-python-tutorial.readthedocs.io/Steric_height.html


# ===== Imports =====
import os
import numpy as np
import xarray as xr
from datetime import timedelta
from dask.distributed import Client, LocalCluster
from set_constant import domain_name, face, i, j

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/steric_height_anomaly"
os.makedirs(output_dir, exist_ok=True)

# ========== Open dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)




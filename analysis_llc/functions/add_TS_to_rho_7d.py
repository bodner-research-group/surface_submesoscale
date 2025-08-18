import xarray as xr
import os
from glob import glob
from dask.distributed import Client, LocalCluster
from set_constant import domain_name  # Replace if needed

# ========== Dask cluster setup ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
base_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"
input_dir = os.path.join(base_path, "TSW_24h_avg")
rho_dir = os.path.join(base_path, "rho_weekly")
output_dir = os.path.join(base_path, "rho_Hml_TS_weekly")
os.makedirs(output_dir, exist_ok=True)

# ========== Get all rho_Hml_7d files ==========
rho_files = sorted(glob(os.path.join(rho_dir, "rho_Hml_7d_*.nc")))

for rho_file in rho_files:
    # Extract date tag
    date_tag = os.path.basename(rho_file).replace("rho_Hml_7d_", "").replace(".nc", "")
    print(f"\nProcessing {date_tag}...")

    # Construct matching Theta/Salt file paths
    tt_file = os.path.join(input_dir, f"tt_24h_{date_tag}.nc")
    ss_file = os.path.join(input_dir, f"ss_24h_{date_tag}.nc")

    # Sanity check
    if not os.path.exists(tt_file) or not os.path.exists(ss_file):
        print(f"Missing input T/S files for {date_tag}, skipping.")
        continue

    # Read T/S datasets with Dask
    ds_tt = xr.open_dataset(tt_file, chunks={})
    ds_ss = xr.open_dataset(ss_file, chunks={})

    # Compute 7-day averages
    T_7d = ds_tt["Theta"].mean("time").compute()
    S_7d = ds_ss["Salt"].mean("time").compute()

    # Read rho_Hml_7d file
    ds_rho = xr.open_dataset(rho_file)

    # Add T_7d and S_7d
    ds_combined = ds_rho.assign({
        "T_7d": T_7d,
        "S_7d": S_7d
    })

    # Save to new output path
    out_path = os.path.join(output_dir, f"rho_Hml_TS_7d_{date_tag}.nc")
    ds_combined.to_netcdf(out_path)
    print(f"Saved: {out_path}")

    # Close datasets
    ds_tt.close()
    ds_ss.close()
    ds_rho.close()
    ds_combined.close()

# ========== Close Dask cluster ==========
client.close()
cluster.close()

print("\nAll files written to:", output_dir)

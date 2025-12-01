import xarray as xr
import numpy as np
import os
from glob import glob
import gc

from set_constant import domain_name, face, i, j

# ---------- HML computation function ----------
def compute_Hml(rho_profile, depth_profile, threshold=0.03):
    """
    Compute mixed layer depth based on potential density profile.
    rho_profile: 1D density array over depth
    depth_profile: 1D depth array (positive down)
    """
    rho_surf = rho_profile[0]      # density at surface (z = -0.5 m)
    mask = rho_profile > rho_surf + threshold

    if not np.any(mask):
        return 0.0

    return float(depth_profile[mask].max())


def main():

    # ---------- Input and output paths ----------
    input_dir  = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
    output_dir = input_dir

    # ---------- Find input files ----------
    rho_files = sorted(glob(os.path.join(input_dir, "rho_Hml_TS_daily_*.nc")))
    if len(rho_files) == 0:
        print("❌ No rho files found.")
        return

    # ---------- Load depth from first file ----------
    # All files contain identical depth values in coordinate "k"
    sample = xr.open_dataset(rho_files[0])
    depth = sample["rho_daily"].coords["k"].values  # shape (k,)
    sample.close()

    print(f"Loaded depth with {len(depth)} vertical levels")

    # ---------- Loop through files ----------
    for file in rho_files:

        date_tag = os.path.basename(file)[17:25]  # extract YYYYMMDD
        out_path = os.path.join(output_dir, f"Hml_daily_{date_tag}.nc")

        if os.path.exists(out_path):
            print(f"⏭️  Skipping {date_tag}, output already exists.")
            continue

        print(f"Processing date {date_tag}")

        ds = xr.open_dataset(file)
        rho = ds["rho_daily"]           # (k, j, i)

        # ---------- Compute Hml for each (j, i) grid point ----------
        Hml = xr.apply_ufunc(
            compute_Hml,
            rho,
            depth,
            input_core_dims=[["k"], ["k"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # ---------- Save ----------
        out_ds = xr.Dataset({"Hml_daily": Hml})
        out_ds.to_netcdf(out_path)
        print(f"Saved: {out_path}")

        ds.close()
        del ds, rho, Hml, out_ds
        gc.collect()


# ---------- Entry point ----------
if __name__ == "__main__":
    main()

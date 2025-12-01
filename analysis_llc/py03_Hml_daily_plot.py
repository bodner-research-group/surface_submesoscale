#### Plot daily averaged Hml


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# --- User options ---
from set_constant import domain_name, face, i, j
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
fig_dir  = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Hml_daily"

os.makedirs(fig_dir, exist_ok=True)

def plot_Hml_field(Hml, date_tag, save_path):
    """
    Plot a 2D Hml field.
    """
    plt.figure(figsize=(10, 6))
    img = plt.pcolormesh(Hml, cmap="gist_ncar", shading="auto", vmax = 0, vmin = -900)
    plt.colorbar(img, label="Hml (m)")
    plt.title(f"Mixed Layer Depth — {date_tag}")
    plt.xlabel("i index")
    plt.ylabel("j index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure: {save_path}")


def main():

    # ========== Open LLC4320 Dataset ==========
    ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

    lat = ds1.YC.isel(face=face, i=i, j=j)
    lon = ds1.XC.isel(face=face, i=i, j=j)

    # Find all Hml files
    hml_files = sorted(glob(os.path.join(data_dir, "Hml_daily_*.nc")))
    if len(hml_files) == 0:
        print("❌ No Hml files found.")
        return

    print(f"Found {len(hml_files)} Hml files")

    for file in hml_files:
        # Get date tag
        date_tag = os.path.basename(file)[10:18]  # "Hml_daily_YYYYMMDD.nc"

        print(f"📅 Plotting {date_tag}")

        # Load dataset
        ds = xr.open_dataset(file)
        Hml = ds["Hml_daily"]       # shape (j, i)

        # Plot & save
        save_path = os.path.join(fig_dir, f"Hml_daily_{date_tag}.png")
        plot_Hml_field(Hml, date_tag, save_path)

        ds.close()


if __name__ == "__main__":
    main()

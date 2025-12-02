import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from matplotlib.animation import FFMpegWriter

# =========================================================
# User options
# =========================================================
from set_constant import domain_name, face, i, j
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
fig_dir  = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/Hml_daily_compare"
os.makedirs(fig_dir, exist_ok=True)

# =========================================================
# Plot function
# =========================================================
def plot_three_panel(Hml_left, Hml_right, lat, lon, date_tag, save_path):

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    # -----------------------------------------------------
    # LEFT — 10 m reference
    # -----------------------------------------------------
    im1 = axs[0].pcolormesh(
        lon, lat, -Hml_left,
        cmap="gist_ncar", shading="auto", vmin=0, vmax=850
    )
    axs[0].set_title(f"Δρ=0.03 relative to surface 0.5m")
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axs[0])

    # -----------------------------------------------------
    # MIDDLE — surface reference
    # -----------------------------------------------------
    im2 = axs[1].pcolormesh(
        lon, lat, -Hml_right,
        cmap="gist_ncar", shading="auto", vmin=0, vmax=850
    )
    axs[1].set_title("Δρ=0.03 relative to 10m")
    axs[1].set_xlabel("Longitude")
    axs[1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axs[1])

    # -----------------------------------------------------
    # RIGHT — difference
    # -----------------------------------------------------
    diff = -Hml_left - (-Hml_right)

    im3 = axs[2].pcolormesh(
        lon, lat, diff,
        cmap="RdBu", shading="auto", vmin=-60, vmax=60
    )
    axs[2].set_title("Difference (surface − 10m)")
    axs[2].set_xlabel("Longitude")
    axs[2].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axs[2], label="Difference (m)")

    fig.suptitle(f"Mixed Layer Depth Comparison — {date_tag}", fontsize=16)

    fig.savefig(save_path, dpi=150)
    plt.close()
    print("Saved:", save_path)


# =========================================================
# Main
# =========================================================
def main():

    # Load full 2D lat/lon grid for this face
    ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
    lat = ds1.YC.isel(face=face, i=i, j=j)
    lon = ds1.XC.isel(face=face, i=i, j=j)

    # File lists
    new_Hml_files = sorted(glob(os.path.join(data_dir, "Hml_daily_surface_reference_*.nc")))
    raw_Hml_files = sorted(glob(os.path.join(data_dir, "rho_Hml_TS_daily_*.nc")))

    print(f"Found {len(new_Hml_files)} processed Hml files")
    print(f"Found {len(raw_Hml_files)} raw TS Hml files")

    images = []

    for file in new_Hml_files:

        date_tag = os.path.basename(file)[10:18]

        raw_file = f"{data_dir}/rho_Hml_TS_daily_{date_tag}.nc"
        if not os.path.exists(raw_file):
            print("Missing raw file for", date_tag)
            continue

        # LEFT (10 m reference)
        ds_left = xr.open_dataset(file)
        Hml_left = ds_left["Hml_daily"].values   # (j,i)

        # MIDDLE (surface reference)
        ds_right = xr.open_dataset(raw_file)
        Hml_right = ds_right["Hml_daily"].values   # (j,i)

        save_path = os.path.join(fig_dir, f"Hml_threepanel_{date_tag}.png")

        plot_three_panel(Hml_left, Hml_right, lat, lon, date_tag, save_path)
        images.append(save_path)

        ds_left.close()
        ds_right.close()

    # =====================================================
    # Make MP4 animation
    # =====================================================
    print("\n🎞️ Creating MP4 animation...")

    mp4_path = os.path.join(fig_dir, "Hml_threepanel_animation.mp4")
    writer = FFMpegWriter(fps=5)

    fig = plt.figure(figsize=(20, 6))

    with writer.saving(fig, mp4_path, dpi=150):
        for img_file in images:
            img = plt.imread(img_file)
            plt.imshow(img)
            plt.axis("off")
            writer.grab_frame()
            plt.clf()

    plt.close()
    print("MP4 saved:", mp4_path)


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    main()

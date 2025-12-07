import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from set_constant import domain_name, face, i, j

# ============================================================
# Directories
# ============================================================
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_daily"
grid_dir = "/orcd/data/abodner/003/LLC4320/LLC4320"
figdir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_mld_daily"
os.makedirs(figdir, exist_ok=True)

# ============================================================
# Load grid
# ============================================================
ds_grid = xr.open_zarr(grid_dir, consolidated=False)
lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)
Lon, Lat = xr.broadcast(lon, lat)

# ============================================================
# Get all daily NetCDF files
# ============================================================
nc_files = sorted(glob(os.path.join(data_dir, "wb_mld_daily_*.nc")))

# ============================================================
# STEP 1 — Scan data to determine global color limits
# ============================================================
print("Scanning NetCDF files to determine global color limits...")

w_vals, wb_vals, wb_fact_vals, wb_eddy_vals, Hml_vals = [], [], [], [], []

for nc in nc_files:
    ds = xr.open_dataset(nc)
    w_vals.append(ds["w_avg"])
    wb_vals.append(ds["wb_avg"])
    wb_fact_vals.append(ds["wb_fact"])
    Hml_vals.append(ds["Hml"])
    wb_eddy_vals.append(ds["wb_avg"] - ds["wb_fact"])
    ds.close()

# Concatenate along fake time dimension
w_all       = xr.concat(w_vals, dim="time_scatter")
wb_all      = xr.concat(wb_vals, dim="time_scatter")
wb_fact_all = xr.concat(wb_fact_vals, dim="time_scatter")
wb_eddy_all = xr.concat(wb_eddy_vals, dim="time_scatter")
Hml_all     = xr.concat(Hml_vals, dim="time_scatter")

# Symmetric limits for zero-centered fields
def sym_limits(da):
    m = float(np.nanmax(np.abs(da.values)))
    return -m, m

w_vmin, w_vmax           = sym_limits(w_all)
wb_vmin, wb_vmax         = sym_limits(wb_all)
wb_fact_vmin, wb_fact_vmax = sym_limits(wb_fact_all)
wb_eddy_vmin, wb_eddy_vmax = sym_limits(wb_eddy_all)

# Non-symmetric Hml limits
Hml_vmin = float(Hml_all.min())
Hml_vmax = float(Hml_all.max())

print("Global color limits determined.")

# ============================================================
# STEP 2 — Plotting function for one day
# ============================================================
def make_daily_plot(nc_path):
    date_tag = os.path.basename(nc_path).split("_")[-1].replace(".nc", "")
    save_path = os.path.join(figdir, f"wb_mld_daily_{date_tag}.png")

    if os.path.exists(save_path):
        print(f"Skipping existing: {save_path}")
        return save_path

    ds = xr.open_dataset(nc_path)
    wb_avg  = ds["wb_avg"]
    w_avg   = ds["w_avg"]
    b_avg   = ds["b_avg"]
    wb_fact = ds["wb_fact"]
    Hml     = ds["Hml"]
    wb_eddy = wb_avg - wb_fact
    ds.close()

    # Fields and color limits
    fields = [
        (w_avg,     r"$\overline{w}^z$",              w_vmin/10, w_vmax/10, "RdBu_r"),
        (b_avg,     r"$\overline{b}^z$",              None, None, "viridis"),
        (Hml,       r"$H_{ml}$",                      Hml_vmin, Hml_vmax, "gist_ncar"),
        (wb_avg,    r"$\overline{wb}^z$",             wb_vmin/10, wb_vmax/10, "RdBu_r"),
        (wb_fact,   r"$\overline{w}^z\overline{b}^z$", wb_fact_vmin/10, wb_fact_vmax/10, "RdBu_r"),
        (wb_eddy,   r"$\overline{wb}^z - \overline{w}^z\overline{b}^z$", wb_eddy_vmin/10, wb_eddy_vmax/10, "RdBu_r")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (field, title, vmin, vmax, cmap) in zip(axes.flatten(), fields):
        kw = {"cmap": cmap}
        if vmin is not None:
            kw["vmin"] = vmin
            kw["vmax"] = vmax
        p = ax.pcolormesh(Lon, Lat, field, **kw)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(p, ax=ax)

    fig.suptitle(f"MLD Vertical Averages — {date_tag}", fontsize=18)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")
    return save_path

# ============================================================
# STEP 3 — Parallel plotting of daily figures
# ============================================================
print("Generating daily 6-panel plots in parallel...")

n_workers = 8  # adjust to your machine
png_files = []

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    for png in executor.map(make_daily_plot, nc_files):
        png_files.append(png)

png_files = sorted(png_files)
print("All daily plots complete.")

# ============================================================
# STEP 4 — Create animation
# ============================================================
print("Creating animation...")

##### Convert images to video
import os
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_mld_daily"
# high-resolution
output_movie = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/movie-wb_mld_daily.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/wb_mld_daily_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")



# sample = plt.imread(png_files[0])

# fig = plt.figure()
# img = plt.imshow(sample, animated=True, aspect='equal')
# plt.axis("off")

# def update(frame):
#     fname = png_files[frame]
#     img.set_array(plt.imread(fname))
#     # plt.title(os.path.basename(fname).replace(".png", ""), fontsize=14)
#     return [img]

# ani = animation.FuncAnimation(fig, update, frames=len(png_files), interval=100)

# mp4_path = os.path.join(figdir, "wb_mld_daily_animation.mp4")
# ani.save(mp4_path, writer="ffmpeg", dpi=150)
# plt.close()

print(f"Animation saved → {mp4_path}")

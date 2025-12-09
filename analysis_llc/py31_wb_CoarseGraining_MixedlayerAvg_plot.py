import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from set_constant import domain_name, face, i, j

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# ============================================================
# Directories
# ============================================================
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_daily"
grid_dir = "/orcd/data/abodner/003/LLC4320/LLC4320"
figdir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/wb_mld_daily"
os.makedirs(figdir, exist_ok=True)

# ============================================================
#                      COARSE-GRAINING FUNCTION
# ============================================================
def coarse_grain(data, window_size=4):
    """
    Coarse-grains the input data by averaging over a window of size (window_size, window_size) in the horizontal plane.
    """
    return data.coarsen(i=window_size, j=window_size, boundary="trim").mean()


# ============================================================
# Load grid
# ============================================================
ds_grid = xr.open_zarr(grid_dir, consolidated=False)
lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)
# Lon, Lat = xr.broadcast(lon, lat)

lat_cg = coarse_grain(lat, window_size=4)
lon_cg = coarse_grain(lon, window_size=4)

# ============================================================
# Load all daily nc files
# ============================================================
nc_files = sorted(glob(os.path.join(data_dir, "wb_mld_daily_*.nc")))
print(f"Found {len(nc_files)} daily files.")

# ============================================================
# STEP 1 — Determine global color limits
# ============================================================
print("\nScanning files for global color limits...")

w_vals, b_vals, wb_vals, wb_fact_vals, B_eddy_vals, Hml_vals = [], [], [], [], [], []

for f in nc_files:
    ds = xr.open_dataset(f)
    w_vals.append(ds["w_avg"])
    b_vals.append(ds["b_avg"])
    wb_vals.append(ds["wb_avg"])
    wb_fact_vals.append(ds["wb_fact"])
    B_eddy_vals.append(ds["B_eddy"])
    Hml_vals.append(ds["Hml_cg"])
    ds.close()

# Combine all
w_all        = xr.concat(w_vals,        dim="time_scatter")
b_all        = xr.concat(b_vals,        dim="time_scatter")
wb_all       = xr.concat(wb_vals,       dim="time_scatter")
wb_fact_all  = xr.concat(wb_fact_vals,  dim="time_scatter")
B_eddy_all   = xr.concat(B_eddy_vals,   dim="time_scatter")
Hml_all      = xr.concat(Hml_vals,      dim="time_scatter")

def sym_limits(da):
    m = float(np.nanmax(np.abs(da)))
    return -m, m

w_vmin, w_vmax             = sym_limits(w_all)
b_vmin, b_vmax             = sym_limits(b_all)
wb_vmin, wb_vmax           = sym_limits(wb_all)
wb_fact_vmin, wb_fact_vmax = sym_limits(wb_fact_all)
B_eddy_vmin, B_eddy_vmax   = sym_limits(B_eddy_all)

Hml_vmin = float(Hml_all.min())
Hml_vmax = float(Hml_all.max())

print(" → Color limits computed.\n")

# ============================================================
# STEP 2 — Plotting function for one snapshot
# ============================================================
def make_daily_plot(nc_path):
    date_tag = os.path.basename(nc_path).split("_")[-1].replace(".nc", "")
    save_file = os.path.join(figdir, f"wb_mld_daily_{date_tag}.png")

    if os.path.exists(save_file):
        print(f"Exists → {save_file}")
        return save_file

    ds = xr.open_dataset(nc_path)

    w_avg    = ds["w_avg"]
    b_avg    = ds["b_avg"]
    wb_avg   = ds["wb_avg"]
    wb_fact  = ds["wb_fact"]
    B_eddy   = ds["B_eddy"]
    Hml_cg   = ds["Hml_cg"]

    ds.close()

    nfac = 15
    # Define 6-panel fields
    fields = [
        (w_avg,    r"$\overline{\overline{w}^{xy}}^z$",        w_vmin/nfac,        w_vmax/nfac,        "RdBu_r"),
        (b_avg,    r"$\overline{\overline{b}^{xy}}^z$",        None, None, "viridis"),
        (Hml_cg,   r"$\overline{H_{ml}}^{xy}$",              Hml_vmin,      Hml_vmax,      "gist_ncar"),
        (wb_avg,   r"$\overline{\overline{wb}^{xy}}^z$",       wb_vmin/nfac,       wb_vmax/nfac,       "RdBu_r"),
        (wb_fact,  r"$\overline{\overline{w}^{xy}}^z\overline{\overline{b}^{xy}}^z$", wb_fact_vmin/nfac, wb_fact_vmax/nfac, "RdBu_r"),
        (B_eddy,   r"$\overline{\overline{wb}^{xy}}^z-\overline{\overline{w}^{xy}}^z\overline{\overline{b}^{xy}}^z$", B_eddy_vmin/nfac, B_eddy_vmax/nfac, "RdBu_r"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (fld, ttl, vmin, vmax, cmap) in zip(axes.flatten(), fields):
        im = ax.pcolormesh(lon_cg, lat_cg, fld, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=14)
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"Coarse-Grained MLD-Averaged Fields — {date_tag}", fontsize=18)
    plt.savefig(save_file, dpi=150)
    plt.close()

    print(f"Saved → {save_file}")
    return save_file

# ============================================================
# STEP 3 — Parallel plot generation
# ============================================================
print("Generating daily plots...\n")
png_files = []

with ProcessPoolExecutor(max_workers=8) as ex:
    for result in ex.map(make_daily_plot, nc_files):
        png_files.append(result)

png_files = sorted(png_files)
print("\nAll plots complete.\n")

# ============================================================
# STEP 4 — Make movie
# ============================================================
print("Creating video...")


##### Convert images to video
import os
output_movie = f"{figdir}/movie-wb_mld_daily.mp4"
os.system(f"ffmpeg -r 15 -pattern_type glob -i '{figdir}/wb_mld_daily_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")

print(f"Movie saved → {output_movie}")

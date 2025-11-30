#!/usr/bin/env python3
"""
Plot monthly means of Eta, U, and V (South Ocean region).
Creates one figure per month with 3 panels: Eta, U, V.
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# ========================= PATHS =========================
# Directory that contains eta_monthly.nc, u_monthly.nc, v_monthly.nc
INPUT_DIR = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/Southern_Ocean_JunyangGou/"

# Directory for output plots
PLOT_DIR = os.path.join(INPUT_DIR, "monthly_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Filenames
ETA_FILE = os.path.join(INPUT_DIR, "eta_monthly.nc")
U_FILE   = os.path.join(INPUT_DIR, "u_monthly.nc")
V_FILE   = os.path.join(INPUT_DIR, "v_monthly.nc")

# ====================== LOAD DATA ========================
print("Loading datasets...")
ds_eta = xr.open_dataset(ETA_FILE)
ds_u   = xr.open_dataset(U_FILE)
ds_v   = xr.open_dataset(V_FILE)

# Verify that all files have same monthly time coordinates
time = ds_eta.time.values

# ===================== PLOTTING HELPERS ==================
def make_map(ax):
    # """Add coastlines, gridlines, etc."""
    # ax.coastlines(resolution="110m", linewidth=0.6)
    # ax.add_feature(cfeature.LAND, facecolor="lightgray")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False

def plot_field(ax, lon, lat, field, title, cmap, vmin=None, vmax=None):
    """Scatter plot of the monthly field."""
    sc = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax,
                   transform=ccrs.PlateCarree(), shading="auto")

    ax.set_title(title, fontsize=14)
    return sc

# ================== COLOR LIMITS (edit if needed) ===================
eta_clim = (-2, 2)   # meters
u_clim   = (-1.0, 1.0)  # m/s
v_clim   = (-1.0, 1.0)  # m/s

# ======================= MAIN LOOP =======================
print("Starting monthly plotting...")

for n, t in enumerate(time):

    YYYY_MM = str(np.datetime_as_string(t, unit="D"))[:7]
    print(f"Plotting month: {YYYY_MM}")

    # Extract monthly arrays
    eta = ds_eta["Eta"].isel(time=n).values
    U   = ds_u["U"].isel(time=n).values
    V   = ds_v["V"].isel(time=n).values

    lon = ds_eta["lon"].values
    lat = ds_eta["lat"].values

    # ---------- FIGURE ----------
    fig = plt.figure(figsize=(12, 13))

    proj = ccrs.SouthPolarStereo(central_longitude=30)

    # Eta
    ax1 = fig.add_subplot(3, 1, 1, projection=proj)
    make_map(ax1)
    sc1 = plot_field(ax1, lon, lat, eta, "Eta (m)", cmap="coolwarm",
                     vmin=eta_clim[0], vmax=eta_clim[1])
    cb1 = plt.colorbar(sc1, ax=ax1, orientation="vertical", pad=0.04, shrink=0.6, fraction=0.05)

    # U
    ax2 = fig.add_subplot(3, 1, 2, projection=proj)
    make_map(ax2)
    sc2 = plot_field(ax2, lon, lat, U, "Surface U (m/s)", cmap="RdBu_r",
                     vmin=u_clim[0], vmax=u_clim[1])
    cb2 = plt.colorbar(sc2, ax=ax2, orientation="vertical", pad=0.04, shrink=0.6, fraction=0.05)


    # V
    ax3 = fig.add_subplot(3, 1, 3, projection=proj)
    make_map(ax3)
    sc3 = plot_field(ax3, lon, lat, V, "Surface V (m/s)", cmap="RdBu_r",
                     vmin=v_clim[0], vmax=v_clim[1])
    cb3 = plt.colorbar(sc3, ax=ax3, orientation="vertical", pad=0.04, shrink=0.6, fraction=0.05)

    fig.suptitle(f"Monthly Surface Fields – {YYYY_MM}", fontsize=16)

    # ---------- SAVE ----------
    outfile = os.path.join(PLOT_DIR, f"monthly_surface_{YYYY_MM}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

print("All monthly plots created.")

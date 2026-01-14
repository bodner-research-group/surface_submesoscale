"""
Complete script: 3D Turner PDF vectors vs mean |∂η| per TuH bin
- lighter background
- equal axis scaling (box aspect)
- scientific notation for axis tick labels
- constant density (isopycnal) lines drawn on the z=0 plane
- saves figure to figdir

REQUIREMENTS / ASSUMPTIONS
- `nc_file`, `figdir`, `bin_centers`, and `mean_deta_per_bin` are defined / available before running.
  * bin_centers is expected to match the TuH bin centers you used (e.g. np.linspace(-180,180,37))
  * mean_deta_per_bin is an array of same length as bin_centers (NaNs allowed)
- xarray, numpy, matplotlib, scipy are installed.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------
# User-provided variables (make sure these exist)
# ---------------------------
from set_constant import domain_name, face, i, j  

# Path to saved output file from previous script
nc_file = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TurnerAngle_3D/Turner_3D_7d_2011-11-01.nc"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/TurnerAngle_3D"
os.makedirs(figdir, exist_ok=True)

bins = np.linspace(-180, 180, 73)  # 5-degree bins
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# mean_deta_per_bin = ...  # same length as bin_centers

# ---------------------------
# Config
# ---------------------------
scale = 1.0             # overall scale factor for vector lengths (tune to taste)
horizontal_scale = 20   # keep similar to original code
vertical_scale = 20
pdf_h_mult = 1.0        # extra multiplier for horizontal PDF vectors (if needed)
pdf_v_mult = 0.5        # extra multiplier for vertical PDF vectors (if needed)
os.makedirs(figdir, exist_ok=True)

# Global font size for plots
plt.rcParams.update({'font.size': 13})

# ---------------------------
# Load dataset
# ---------------------------
ds_pdf = xr.open_dataset(nc_file)

x_grid = ds_pdf["x_grid"].data  # angle bins in degrees
pdf_values_h = ds_pdf["pdf_values_h"].data
pdf_values_v = ds_pdf["pdf_values_v"].data
alpha = np.nanmean(ds_pdf["alpha_surf"].data)
beta = np.nanmean(ds_pdf["beta_surf"].data)

# ---------------------------
# Unit vectors (projection space)
# ---------------------------
slope_rho = 1.0
v_cross = np.array([-slope_rho, 1.0])
v_iso = np.array([1.0, slope_rho])
v_cross /= np.linalg.norm(v_cross)
v_iso /= np.linalg.norm(v_iso)

# ---------------------------
# Interpolate mean |∂η| onto x_grid (PDF bins)
# ---------------------------
valid = ~np.isnan(mean_deta_per_bin)
if valid.sum() < 2:
    # Not enough points to interpolate -> fill with nan
    z_vals = np.full_like(x_grid, np.nan, dtype=float)
else:
    interp_func = interp1d(bin_centers[valid], mean_deta_per_bin[valid],
                           kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=False)
    z_vals = interp_func(x_grid)

# ---------------------------
# Compute projected (x,y) for each angle using TuH magnitudes
# ---------------------------
x_proj = []
y_proj = []
for angle_deg, mag_h in zip(x_grid, pdf_values_h):
    dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
    dx = mag_h * dir_vec[0] * beta * scale * horizontal_scale * pdf_h_mult
    dy = mag_h * dir_vec[1] * alpha * scale * horizontal_scale * pdf_h_mult
    x_proj.append(dx)
    y_proj.append(dy)
x_proj = np.array(x_proj, dtype=float)
y_proj = np.array(y_proj, dtype=float)

# ---------------------------
# Prepare 3D figure (lighter background)
# ---------------------------
fig = plt.figure(figsize=(11, 9), facecolor="#fbfbfb")  # very light gray background
ax = fig.add_subplot(111, projection='3d')

# Make 3D axes panes lighter (white-ish)
pane_color = (0.98, 0.98, 0.98, 1.0)
ax.xaxis.set_pane_color(pane_color)
ax.yaxis.set_pane_color(pane_color)
ax.zaxis.set_pane_color(pane_color)
# Also set edge colors (optional subtle grid)
for spine_name in ('left', 'right', 'top', 'bottom'):
    try:
        getattr(ax, f"{spine_name}_axis")
    except Exception:
        pass

# ---------------------------
# Compute equal axis scaling and limits
# ---------------------------
# Use only the finite coords
finite_mask = np.isfinite(x_proj) & np.isfinite(y_proj) & np.isfinite(z_vals)
if finite_mask.sum() == 0:
    # fallback if no finite vector endpoints: build small default ranges around zero
    x_vals_all = np.array([0.0, 1.0])
    y_vals_all = np.array([0.0, 1.0])
    z_vals_all = np.array([0.0, 1.0])
else:
    x_vals_all = np.concatenate([x_proj[finite_mask], [0]])
    y_vals_all = np.concatenate([y_proj[finite_mask], [0]])
    z_vals_all = np.concatenate([z_vals[finite_mask], [0]])

x_min, x_max = np.nanmin(x_vals_all), np.nanmax(x_vals_all)
y_min, y_max = np.nanmin(y_vals_all), np.nanmax(y_vals_all)
z_min, z_max = np.nanmin(z_vals_all), np.nanmax(z_vals_all)

# Add a little padding
pad_frac = 0.08
x_range = x_max - x_min if (x_max - x_min) != 0 else 1.0
y_range = y_max - y_min if (y_max - y_min) != 0 else 1.0
z_range = z_max - z_min if (z_max - z_min) != 0 else 1.0

xlim = (x_min - pad_frac * x_range, x_max + pad_frac * x_range)
ylim = (y_min - pad_frac * y_range, y_max + pad_frac * y_range)
zlim = (z_min - pad_frac * z_range, z_max + pad_frac * z_range)

# For equal scaling, set box aspect to 1,1,1 and center limits around midpoints with same half-range
max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
x_mid = 0.5 * (xlim[0] + xlim[1])
y_mid = 0.5 * (ylim[0] + ylim[1])
z_mid = 0.5 * (zlim[0] + zlim[1])

half = 0.5 * max_range + 1e-16
ax.set_xlim(x_mid - half, x_mid + half)
ax.set_ylim(y_mid - half, y_mid + half)
ax.set_zlim(z_mid - half, z_mid + half)

# Use set_box_aspect for correct equal aspect (matplotlib >= 3.3)
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    # older matplotlib - we already set symmetric limits which approximates equal scaling
    pass

# ---------------------------
# Normalize z values for colormap (handle NaNs)
# ---------------------------
z_finite = z_vals[np.isfinite(z_vals)]
if z_finite.size > 0:
    zmin_plot = np.nanmin(z_vals)
    zmax_plot = np.nanmax(z_vals)
    if zmax_plot == zmin_plot:
        zmax_plot = zmin_plot + 1.0
    norm = Normalize(vmin=zmin_plot, vmax=zmax_plot)
else:
    # fallback
    zmin_plot, zmax_plot = 0.0, 1.0
    norm = Normalize(vmin=zmin_plot, vmax=zmax_plot)

cmap = plt.cm.plasma
mappable = ScalarMappable(norm=norm, cmap=cmap)

# ---------------------------
# Plot 3D lines from origin to (x_proj, y_proj, z_vals)
# ---------------------------
for x, y, z in zip(x_proj, y_proj, z_vals):
    if np.isfinite(z) and np.isfinite(x) and np.isfinite(y):
        color = cmap(norm(z))
        ax.plot([0.0, x], [0.0, y], [0.0, z], color=color, alpha=0.9, linewidth=1.25)

# ---------------------------
# Dummy scatter for colorbar
# ---------------------------
# We create a scatter of the z-values at the endpoints in order to create a mappable for the colorbar
endpoints_mask = np.isfinite(x_proj) & np.isfinite(y_proj) & np.isfinite(z_vals)
ax.scatter(x_proj[endpoints_mask], y_proj[endpoints_mask], z_vals[endpoints_mask],
           c=z_vals[endpoints_mask], cmap=cmap, norm=norm, s=10, alpha=0.8)

# ---------------------------
# Plot TuH (horizontal PDF) vectors on z=0 plane (dashed)
# ---------------------------
for angle_deg, mag_h in zip(x_grid, pdf_values_h):
    dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
    dx = mag_h * dir_vec[0] * beta * scale * horizontal_scale * pdf_h_mult
    dy = mag_h * dir_vec[1] * alpha * scale * horizontal_scale * pdf_h_mult
    # dashed gray on z=0 plane
    ax.plot([0, dx], [0, dy], [0, 0], color='darkgray', linestyle='--', alpha=0.7, linewidth=0.8)

# ---------------------------
# Plot TuV (vertical PDF) vectors on z=0 plane (solid green)
# ---------------------------
for angle_deg, mag_v in zip(x_grid, pdf_values_v):
    dir_vec = np.cos(np.deg2rad(angle_deg)) * v_cross + np.sin(np.deg2rad(angle_deg)) * v_iso
    dx = mag_v * dir_vec[0] * beta * scale * horizontal_scale * pdf_v_mult
    dy = mag_v * dir_vec[1] * alpha * scale * horizontal_scale * pdf_v_mult
    ax.plot([0, dx], [0, dy], [0, 0], color='green', alpha=0.75, linewidth=0.8)

# ---------------------------
# Add constant density (isopycnal) lines on z=0 plane
#    Use the same approach as your 2D code: T = slope_rho * S + c
# ---------------------------
# Set S_line across the x-range (x-axis corresponds to β∂S projection space)
S_line = np.linspace(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1, 400)
# choose c offsets to cover the y-range
c_min = ax.get_ylim()[0] * 1.1 - slope_rho * S_line.min()
c_max = ax.get_ylim()[1] * 1.1 - slope_rho * S_line.max()
# choose a modest number of lines
n_lines = 16
c_values = np.linspace(c_min, c_max, n_lines)

for c in c_values:
    T_line = slope_rho * S_line + c
    # plot on z=0 plane with light gray and low alpha
    ax.plot(S_line, T_line, zs=0.0, zdir='z', color='gray', linewidth=0.45, alpha=0.25)

# ---------------------------
# Reference axes labels, title, and colorbar
# ---------------------------
ax.set_xlabel(r"$\beta \, \partial S$ (PDF projection)")
ax.set_ylabel(r"$\alpha \, \partial \theta$ (PDF projection)")
ax.set_zlabel(r"Mean $|\partial \eta|$")

ax.set_title("3D Turner PDF Vectors vs. Mean $|\\partial \\eta|$", fontsize=15, pad=12)

# colorbar for mean |∂η|
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, fraction=0.06)
cbar.set_label(r"Mean $|\partial \eta|$")

# ---------------------------
# Scientific notation for tick labels on all axes
# ---------------------------
scalar_formatter = ticker.ScalarFormatter(useMathText=True)
scalar_formatter.set_powerlimits((-4, 4))  # switch to scientific notation outside this range
ax.xaxis.set_major_formatter(scalar_formatter)
ax.yaxis.set_major_formatter(scalar_formatter)
ax.zaxis.set_major_formatter(scalar_formatter)

# For 3D Axes the tick label formatting sometimes requires accessing the axis attribute objects:
try:
    ax.xaxis._axinfo["tick"]["formatter"] = scalar_formatter
    ax.yaxis._axinfo["tick"]["formatter"] = scalar_formatter
    ax.zaxis._axinfo["tick"]["formatter"] = scalar_formatter
except Exception:
    pass

# Optionally show grid (subtle)
ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)

# Add small legend markers manually (we'll add simple 2D legend via scatter proxies)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='darkgray', lw=1, linestyle='--', label=r'$\mathrm{Tu}_H$ (proj, z=0)'),
    Line2D([0], [0], color='green', lw=1, label=r'$\mathrm{Tu}_V$ (proj, z=0)'),
    Line2D([0], [0], color='gray', lw=0.6, alpha=0.5, label='isopycnals (z=0)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()

# ---------------------------
# Save & close
# ---------------------------
fig_path = os.path.join(figdir, "3d_turner_pdf_with_lines.png")
plt.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)
ds_pdf.close()

print("✅ 3D Turner PDF figure saved to:", fig_path)
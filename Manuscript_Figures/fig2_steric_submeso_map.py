import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ==============================================================
# Domain
# ==============================================================
domain_name = "icelandic_basin"

# ==============================================================
# Paths
# ==============================================================
data_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/Manuscript_Data/{domain_name}/steric_submeso"
figdir = data_dir
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Choose day
# ==============================================================
# date_tag = "20120215"
date_tag = "20120305"
datafile = os.path.join(data_dir, f"grad_laplace_eta_submeso_{date_tag}_16.50kmCutoff.nc")

ds = xr.open_dataset(datafile)

lon = ds.lon
lat = ds.lat

# ==== Compute Coriolis parameter (mean) ====
omega = 7.2921e-5  # [rad/s]
lat_rad = np.deg2rad(lat)
f0 = 2 * omega * np.sin(lat_rad)
f0_mean = f0.mean().compute()
print(f"Mean f0 over domain: {f0_mean:.2e} s^-1")

gravity = 9.81

# ==============================================================
# Plot settings
# ==============================================================
plt.rcParams.update({"font.size": 19})
plt.rcParams.update({"axes.titlesize": 25})

cmap_rb = "RdBu_r"
cmap_grad = "viridis"

# ssh_lim = 0.2
# grad_lim = 5e-6 *gravity/f0_mean
# lap_lim = 1e-9 *gravity/(f0_mean**2)
ssh_lim = 0.15
grad_lim = 0.35
lap_lim = 0.6

lon_min, lon_max = -27.0, -17.5
lat_min, lat_max = 57.9, 62.3


# ==============================================================
# Figure
# ==============================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)
# fig, axes = plt.subplots(3, 3, figsize=(18, 16))

fig.set_constrained_layout_pads(
    w_pad=0.1, h_pad=0.1,
    wspace=0.05, hspace=0.05
)

# fig.subplots_adjust(
#     wspace=0.25,   # horizontal space between panels
#     hspace=0.30    # vertical space between panels
# )

# fig.suptitle(f"SSH diagnostics — {date_tag}", fontsize=33)

# import matplotlib as mpl

# mpl.rcParams['mathtext.fontset'] = 'cm'   # Computer Modern
# mpl.rcParams['font.family'] = 'serif'

# ==============================================================
# -------- Row 1: SSH fields --------
# ==============================================================
im00 = axes[0, 0].pcolormesh(
    lon, lat, ds.eta_minus_mean.squeeze(),
    cmap=cmap_rb, shading="auto",
    vmin=-ssh_lim, vmax=ssh_lim
)
axes[0, 0].set_title(r"SSH anomaly $\eta-\langle \eta \rangle$")
axes[0, 0].set_ylabel("Latitude")
# axes[0, 0].text(0.02, 0.96, "(a)", transform=axes[0, 0].transAxes,
                # fontweight="bold", va="top")

im01 = axes[0, 1].pcolormesh(
    lon, lat, ds.eta_steric.squeeze(),
    cmap=cmap_rb, shading="auto",
    vmin=-ssh_lim, vmax=ssh_lim
)
axes[0, 1].set_title(r"Steric height $\eta_{\mathrm{steric}}$")

im02 = axes[0, 2].pcolormesh(
    lon, lat, ds.eta_submeso.squeeze(),
    cmap=cmap_rb, shading="auto",
    vmin=-ssh_lim, vmax=ssh_lim
)
axes[0, 2].set_title(r"Submesoscale SSH $\eta_{\mathrm{submeso}}$")

cbar0 = fig.colorbar(
    im02, ax=axes[0, :],
    orientation="vertical", fraction=0.032, shrink=0.85, pad=0.02
)

# ==============================================================
# -------- Row 2: Gradient magnitude --------
# ==============================================================
im10 = axes[1, 0].pcolormesh(
    lon, lat, ds.eta_grad_mag.squeeze() *gravity/f0_mean,
    cmap=cmap_grad, shading="auto",
    vmin=0, vmax=grad_lim
)
axes[1, 0].set_title(r"$|\nabla \eta| g/f_0$")
axes[1, 0].set_ylabel("Latitude")

im11 = axes[1, 1].pcolormesh(
    lon, lat, ds.eta_steric_grad_mag.squeeze() *gravity/f0_mean,
    cmap=cmap_grad, shading="auto",
    vmin=0, vmax=grad_lim
)
axes[1, 1].set_title(r"$|\nabla \eta_{\mathrm{steric}}| g/f_0$")

im12 = axes[1, 2].pcolormesh(
    lon, lat, ds.eta_submeso_grad_mag.squeeze() *gravity/f0_mean,
    cmap=cmap_grad, shading="auto",
    vmin=0, vmax=grad_lim
)
axes[1, 2].set_title(r"$|\nabla \eta_{\mathrm{submeso}}| g/f_0$")

cbar1 = fig.colorbar(
    im12, ax=axes[1, :],
    orientation="vertical", fraction=0.032, shrink=0.85, pad=0.02
)

# ==============================================================
# -------- Row 3: Laplacian --------
# ==============================================================
im20 = axes[2, 0].pcolormesh(
    lon, lat, ds.eta_laplace.squeeze()  *gravity/(f0_mean**2),
    cmap=cmap_rb, shading="auto",
    vmin=-lap_lim, vmax=lap_lim
)
axes[2, 0].set_title(r"$\nabla^2 \eta g/f_0^2$")
axes[2, 0].set_xlabel("Longitude")
axes[2, 0].set_ylabel("Latitude")

im21 = axes[2, 1].pcolormesh(
    lon, lat, ds.eta_steric_laplace.squeeze() *gravity/(f0_mean**2),
    cmap=cmap_rb, shading="auto",
    vmin=-lap_lim, vmax=lap_lim
)
axes[2, 1].set_title(r"$\nabla^2 \eta_{\mathrm{steric}} g/f_0^2$")
axes[2, 1].set_xlabel("Longitude")

im22 = axes[2, 2].pcolormesh(
    lon, lat, ds.eta_submeso_laplace.squeeze() *gravity/(f0_mean**2),
    cmap=cmap_rb, shading="auto",
    vmin=-lap_lim, vmax=lap_lim
)
axes[2, 2].set_title(r"$\nabla^2 \eta_{\mathrm{submeso}} g/f_0^2$")
axes[2, 2].set_xlabel("Longitude")

cbar2 = fig.colorbar(
    im22, ax=axes[2, :],
    orientation="vertical", fraction=0.032, shrink=0.85, pad=0.02
)

# ==============================================================
# Apply domain limits & clean labels
# ==============================================================
for ax in axes.flat:
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

for ax in axes[0, :]:
    ax.set_xlabel("")
for ax in axes[1, :]:
    ax.set_xlabel("")
for ax in axes[:, 1:].flat:
    ax.set_ylabel("")

# ==============================================================
# Save
# ==============================================================
outfile = os.path.join(figdir, f"SSH_3x3_grad_laplace_{date_tag}_16.50kmCutoff.png")
plt.savefig(outfile, dpi=300)
plt.close()

print(f"✅ Saved figure: {outfile}")

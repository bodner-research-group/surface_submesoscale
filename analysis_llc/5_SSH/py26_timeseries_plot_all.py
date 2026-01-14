import os
import xarray as xr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16}) # Global font size setting for figures

# ==============================================================
# Paths
# ==============================================================
domain_name = "icelandic_basin"

base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_submesoscale/"
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Files to plot
# ==============================================================
files = {
    "Gaussian 10 km": "SSH_Gaussian_submeso_10kmCutoff_timeseries.nc",
    "Gaussian 20 km": "SSH_Gaussian_submeso_20kmCutoff_timeseries.nc",
    "Gaussian 30 km": "SSH_Gaussian_submeso_30kmCutoff_timeseries.nc",
    "RollingMean 10 km": "SSH_RollingMean_submeso_10kmCutoff_timeseries.nc",
    "RollingMean 20 km": "SSH_RollingMean_submeso_20kmCutoff_timeseries.nc",
    "RollingMean 30 km": "SSH_RollingMean_submeso_30kmCutoff_timeseries.nc",
}

# ==============================================================
# Line style rules
# ==============================================================

# Color by method
colors = {
    "Gaussian": "tab:blue",
    "RollingMean": "tab:green",
}

# Line style by cutoff scale
linestyles = {
    "10 km": "-",
    "20 km": "--",
    "30 km": ":",
}

# ==============================================================
# Create figure
# ==============================================================
plt.figure(figsize=(10, 5))

for label, fname in files.items():
    fpath = os.path.join(base_dir, fname)
    ds = xr.open_dataset(fpath)

    var = ds["eta_submeso_grad2_mean"]

    # Determine color based on method
    method = "Gaussian" if "Gaussian" in label else "RollingMean"
    color = colors[method]

    # Determine line style based on scale
    if "10 km" in label:
        ls = linestyles["10 km"]
    elif "20 km" in label:
        ls = linestyles["20 km"]
    elif "30 km" in label:
        ls = linestyles["30 km"]
    else:
        ls = "-"  # fallback

    plt.plot(
        var["time"],
        var,
        label=label,
        linewidth=1.5,
        linestyle=ls,
        color=color,
    )

# ==============================================================
# Figure formatting
# ==============================================================
plt.title("Comparison of Submesoscale |∇η′|² Timeseries\n(Gaussian vs Rolling Mean Filters)")
plt.xlabel("Time")
plt.ylabel("Mean (submesoscale |∇η′|²) [m²/m²]")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper right", fontsize=9, ncol=2)
plt.tight_layout()

# Save figure
outfile = f"{figdir}SSH_submeso_filter_comparison_timeseries.png"
plt.savefig(outfile, dpi=200)
plt.close()

print(f"✅ Saved combined plot: {outfile}")








################################
################################
################################
#### Plot mesoscales
################################
################################
################################

import os
import xarray as xr
import matplotlib.pyplot as plt

# ==============================================================
# Paths
# ==============================================================
domain_name = "icelandic_basin"

base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale/"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_submesoscale/"
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Mesoscale Gaussian files
# ==============================================================
files = {
    "Gaussian Meso 10 km": "SSH_Gaussian_meso_10kmCutoff_timeseries.nc",
    "Gaussian Meso 20 km": "SSH_Gaussian_meso_20kmCutoff_timeseries.nc",
    "Gaussian Meso 30 km": "SSH_Gaussian_meso_30kmCutoff_timeseries.nc",
}

# Line styles for scales
linestyles = {
    "10 km": "-",
    "20 km": "--",
    "30 km": ":",
}

color = "tab:blue"   # all Gaussian

# ==============================================================
# Plot
# ==============================================================
plt.figure(figsize=(10, 5))

for label, fname in files.items():
    filepath = os.path.join(base_dir, fname)
    ds = xr.open_dataset(filepath)

    # variable name convention (follow your earlier output)
    var = ds["eta_submeso"] if "eta_submeso" in ds else list(ds.data_vars.values())[0]

    # choose linestyle based on km
    if "10 km" in label:
        ls = linestyles["10 km"]
    elif "20 km" in label:
        ls = linestyles["20 km"]
    elif "30 km" in label:
        ls = linestyles["30 km"]
    else:
        ls = "-"

    plt.plot(
        var["time"],
        var,
        label=label,
        linewidth=1.6,
        linestyle=ls,
        color=color,
    )

# ==============================================================
# Format figure
# ==============================================================
plt.title("Mesoscale SSH (Gaussian Filter)\n10 km vs 20 km vs 30 km")
plt.xlabel("Time")
plt.ylabel("Mesoscale SSH [m]")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper right")
plt.tight_layout()

# Save figure
outfile = f"{figdir}SSH_Gaussian_meso_comparison_timeseries.png"
plt.savefig(outfile, dpi=200)
plt.close()

print(f"✅ Saved mesoscale plot: {outfile}")














################################
################################
################################
#### Plot GCM Filters
################################
################################
################################

import os
import xarray as xr
import matplotlib.pyplot as plt

# ==============================================================
# Paths
# ==============================================================
domain_name = "icelandic_basin"

base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_submesoscale"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/SSH_submesoscale/"
os.makedirs(figdir, exist_ok=True)

# ==============================================================
# Files
# ==============================================================
files = {
    "GCMFilters Submeso 10 km": "SSH_GCMFilters_submeso_10kmCutoff_timeseries.nc",
    "GCMFilters Submeso 20 km": "SSH_GCMFilters_submeso_20kmCutoff_timeseries.nc",
    "GCMFilters Submeso 30 km": "SSH_GCMFilters_submeso_30kmCutoff_timeseries.nc",
}

# ==============================================================
# Colors for each scale (different purples)
# ==============================================================
colors = {
    "10 km": "#7b3294",  # dark purple
    "20 km": "#c2a5cf",  # medium lavender
    "30 km": "#e7d4e8",  # light lavender
}

# ==============================================================
# Line styles (optional: keep simple)
# ==============================================================
linestyles = {
    "10 km": "-",
    "20 km": "--",
    "30 km": ":",
}

# ==============================================================
# Plot
# ==============================================================
plt.figure(figsize=(10, 5))

for label, fname in files.items():
    filepath = os.path.join(base_dir, fname)
    ds = xr.open_dataset(filepath)

    # try to determine the correct variable name
    if "eta_submeso" in ds:
        var = ds["eta_submeso"]
    elif "SSH_submeso" in ds:
        var = ds["SSH_submeso"]
    else:
        var = list(ds.data_vars.values())[0]

    # determine scale key (10 km, 20 km, 30 km)
    scale = "10 km" if "10 km" in label else "20 km" if "20 km" in label else "30 km"

    plt.plot(
        var["time"],
        var,
        label=label,
        linewidth=1.8,
        linestyle=linestyles[scale],
        color=colors[scale],
    )

    # ============================
    # 7-day rolling mean (cyan)
    # ============================
    var_smooth = var.rolling(time=7, center=True).mean()

    plt.plot(
        var["time"],
        var_smooth,
        color="cyan",
        linewidth=2.2,
        linestyle="--",
        alpha=0.9,
        label=f"{label} (7-day mean)",
    )

# ==============================================================
# Formatting
# ==============================================================
plt.title("Submesoscale SSH — GCMFilters\n10 km vs 20 km vs 30 km")
plt.xlabel("Time")
plt.ylabel("Submesoscale SSH [m]")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper right")
plt.tight_layout()

# Save figure
outfile = f"{figdir}SSH_GCMFilters_submeso_comparison_timeseries.png"
plt.savefig(outfile, dpi=200)
plt.close()

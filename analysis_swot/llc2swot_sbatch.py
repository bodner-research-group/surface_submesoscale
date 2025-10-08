# ========== DASK SETUP ==========
from dask.distributed import Client, LocalCluster
from dask import delayed, compute
import dask
import gc

# ========== MODULE IMPORTS ==========
import os
import glob
import numpy as np
import xarray as xr
from scipy import interpolate
from inpoly.inpoly2 import inpoly2
import pyinterp
import warnings
from datetime import datetime
import pandas as pd


def main():
    # === Shared Inputs ===
    model_file = "/orcd/data/abodner/003/LLC4320/LLC4320"
    output_all = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/LLC4320/"
    interpolator = "pyinterp_interpolator"  # or "scipy_interpolator"
    model_lat_var = "YC"
    model_lon_var = "XC"
    model_time_var = "time"
    model_ssh_var = "Eta"

    # model_cache_dir = os.path.join(output_all, "../model_cache")
    model_cache_dir = os.path.abspath(os.path.join(output_all, "../model_cache"))
    os.makedirs(model_cache_dir, exist_ok=True)

    # Start Dask cluster
    cluster = LocalCluster(n_workers=17, threads_per_worker=1, memory_limit="22.5GB")
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    print("Loading model dataset...")
    ds_model_all = xr.open_zarr(model_file, consolidated=False)

    # === Load / Cache static grid ===
    lat_cache_path = os.path.join(model_cache_dir, "lat_clean.npy")
    lon_cache_path = os.path.join(model_cache_dir, "lon_clean.npy")
    mask_cache_path = os.path.join(model_cache_dir, "valid_mask.npy")

    if not os.path.exists(lat_cache_path) or not os.path.exists(lon_cache_path):
        print("Caching static model grid (lat/lon)...")
        lat_values = np.concatenate(ds_model_all[model_lat_var], axis=1)
        lon_values = np.concatenate(ds_model_all[model_lon_var], axis=1)

        mask_valid = ~((lon_values.flatten() == 0) & (lat_values.flatten() == 0))
        lon_clean = lon_values.flatten()[mask_valid]
        lat_clean = lat_values.flatten()[mask_valid]

        np.save(lon_cache_path, lon_clean)
        np.save(lat_cache_path, lat_clean)
        np.save(mask_cache_path, mask_valid)
    else:
        print("Loading cached model grid...")
        lon_clean = np.load(lon_cache_path, mmap_mode='r')
        lat_clean = np.load(lat_cache_path, mmap_mode='r')
        mask_valid = np.load(mask_cache_path, mmap_mode='r')

    points = np.column_stack((lon_clean, lat_clean))

    # Model time coordinates
    model_times = pd.to_datetime(ds_model_all[model_time_var].values)

    # === Processing Function ===
    @delayed
    def process_swot_file(swot_file):
        # Load cached grid
        lon_clean = np.load(lon_cache_path, mmap_mode='r')
        lat_clean = np.load(lat_cache_path, mmap_mode='r')
        mask_valid = np.load(mask_cache_path, mmap_mode='r')
        points = np.column_stack((lon_clean, lat_clean))

        fname = os.path.basename(swot_file)

        try:
            time_str_start = fname.split("_")[-3]
            time_str_end = fname.split("_")[-2].split("_v")[0]
            mean_time = datetime.strptime(time_str_start, "%Y%m%dT%H%M%S") + \
                        (datetime.strptime(time_str_end, "%Y%m%dT%H%M%S") - datetime.strptime(time_str_start, "%Y%m%dT%H%M%S")) / 2
            mean_time = pd.to_datetime(np.datetime64(mean_time.replace(minute=0, second=0)))  # round to nearest hour
        except Exception as e:
            print(f"Skipping file due to time parse error: {fname}")
            return

        mask_same_time = (
            (model_times.month == mean_time.month) &
            (model_times.day == mean_time.day) &
            (model_times.hour == mean_time.hour)
        )

        times_2011 = model_times[(model_times.year == 2011) & mask_same_time]
        times_2012 = model_times[(model_times.year == 2012) & mask_same_time]

        model_timestep_index = None
        if len(times_2011) > 0:
            model_timestep_index = np.where(model_times == times_2011[0])[0][0]
        elif len(times_2012) > 0:
            model_timestep_index = np.where(model_times == times_2012[0])[0][0]

        if model_timestep_index is None:
            print(f"No matching model time found for {fname}")
            return
        
        out_fname = "LLC4320_" + fname[len("SWOT_"):]
        # out_fname = "llc2swot_SSH_" + model_times[model_timestep_index].strftime("%Y%m%dT%H") + ".nc"
        out_path = os.path.join(output_dir, out_fname)

        # === Check if output already exists ===
        if os.path.exists(out_path):
            print(f"Output already exists, skipping: {out_path}")
            return

        print(f"\nProcessing file: {fname}")
        print(f"Mean time: {mean_time} → Closest model time: {model_times[model_timestep_index]} (index {model_timestep_index})")

        ds_swot = xr.open_dataset(swot_file, engine="netcdf4")
        ds_model = ds_model_all.isel({model_time_var: model_timestep_index})

        var_values = np.concatenate(ds_model[model_ssh_var].values, axis=1)
        var_clean = var_values.flatten()[mask_valid]
        del var_values, ds_model

        # Polygon creation
        X = xr.where(ds_swot.longitude <= 180, ds_swot.longitude, ds_swot.longitude - 360)
        Y = ds_swot.latitude
        dy = Y.values[-1, 0] - Y.values[0, 0]
        k = 2 if dy > 0 else -2
        k1 = abs(k)

        xx = np.concatenate([
            X.isel(num_lines=0).values,
            X.isel(num_pixels=-1).values + k,
            X.isel(num_lines=-1).values[::-1],
            X.isel(num_pixels=0).values[::-1] - k
        ])
        yy = np.concatenate([
            Y.isel(num_lines=0).values - k,
            Y.isel(num_pixels=-1).values - k1,
            Y.isel(num_lines=-1).values[::-1] + k,
            Y.isel(num_pixels=0).values[::-1] + k1
        ])
        polygon = np.column_stack((xx, yy))
        inside, on_edge = inpoly2(points, polygon)
        mask = inside | on_edge

        del points

        lon_in = lon_clean[mask]
        lat_in = lat_clean[mask]
        var_in = var_clean[mask]

        del lon_clean, lat_clean, var_clean

        if np.size(var_in) == 0:
            warnings.warn(f"No model data found within SWOT swath area for {fname}.")
            return

        # Interpolation
        if interpolator == "scipy_interpolator":
            finterp = interpolate.LinearNDInterpolator(list(zip(lat_in, lon_in)), var_in, fill_value=np.nan)
        elif interpolator == "pyinterp_interpolator":
            points_for_pyinterp = np.column_stack((lon_in, lat_in))
            finterp = pyinterp.RTree()
            finterp.packing(points_for_pyinterp, var_in)
        else:
            raise ValueError(f"Unknown interpolator: {interpolator}")

        lat_swot = ds_swot.latitude.values
        lon_swot = xr.where(ds_swot.longitude > 180, ds_swot.longitude - 360, ds_swot.longitude).values

        if interpolator == "scipy_interpolator":
            points_swot = np.column_stack((lat_swot.flatten(), lon_swot.flatten()))
            ssh_interp = finterp(points_swot).reshape(lat_swot.shape)
        elif interpolator == "pyinterp_interpolator":
            points_swot = np.column_stack((lon_swot.flatten(), lat_swot.flatten()))
            ssh_interp = finterp.inverse_distance_weighting(
                coordinates=points_swot,
                k=5,
                num_threads=0,
                p=2,
                within=True
            )[0].reshape(lat_swot.shape)

        ds_out = xr.Dataset({
            "ssh": (["num_lines", "num_pixels"], ssh_interp)
        }, coords={
            "latitude": (["num_lines", "num_pixels"], lat_swot),
            "longitude": (["num_lines", "num_pixels"], ds_swot.longitude.values)
        })

        # Quality mask
        cross_dist = ds_swot.cross_track_distance
        quality_flag = ds_swot.quality_flag
        mask_dist = xr.where(
            (abs(cross_dist) <= 60.0) & (abs(cross_dist) >= 10.0) & (quality_flag < 101),
            cross_dist,
            np.nan
        )
        ds_out["ssh"] = ds_out["ssh"].where(~np.isnan(mask_dist))
        ds_out.coords['longitude'] = xr.where(ds_out.longitude < 0, ds_out.longitude + 360, ds_out.longitude)

        ds_out.to_netcdf(out_path)

        print(f"Saved: {out_path}")

        del ds_swot, var_in, lon_in, lat_in, ds_out, mask_dist
        gc.collect()
        return out_path

    # === Loop through SWOT cycles ===
    for cycle_num in range(8, 27):
        cycle_dir = f"cycle_{cycle_num:03d}"
        print(f"\n\n=== Processing {cycle_dir} ===")

        swot_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/global_swot_grid_2024/{cycle_dir}"
        global output_dir
        output_dir = os.path.join(output_all, cycle_dir)
        os.makedirs(output_dir, exist_ok=True)

        swot_files = sorted(glob.glob(os.path.join(swot_dir, "SWOT_GRID_L3_LR_SSH_*.nc")))
        print(f"Found {len(swot_files)} SWOT files.")

        if not swot_files:
            print(f"No files found in {swot_dir}. Skipping...")
            continue

        batch_size = 17
        for i in range(0, len(swot_files), batch_size):
            batch = swot_files[i:i+batch_size]
            tasks = [process_swot_file(f) for f in batch]
            compute(*tasks)

    print("\nAll cycles processed.")
    client.shutdown()


if __name__ == "__main__":
    main()

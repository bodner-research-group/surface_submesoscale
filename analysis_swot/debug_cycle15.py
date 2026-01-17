import xarray as xr
import os

# Define the folder containing the NetCDF files
folder = '/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/llc4320_to_swot/cycle_015'
files = [f for f in os.listdir(folder) if f.endswith('.nc')]

# Dictionary to store the file name and the corresponding size of the 'num_lines' dimension
file_dim_sizes = {}

for f in files:
    file_path = os.path.join(folder, f)
    try:
        ds = xr.open_dataset(file_path)
        # Check if 'num_lines' is a dimension in the dataset
        if 'num_lines' in ds.dims:
            size = ds.dims['num_lines']
            file_dim_sizes[f] = size
        else:
            print(f"{f} does not contain the 'num_lines' dimension.")
        ds.close()
    except Exception as e:
        print(f"Error opening file {f}: {e}")

# Identify all unique sizes of the 'num_lines' dimension
sizes = set(file_dim_sizes.values())
print("Unique sizes of the 'num_lines' dimension found in files:", sizes)

# List which files have each size
for size in sizes:
    print(f"Files with num_lines = {size}:")
    for f, s in file_dim_sizes.items():
        if s == size:
            print("  ", f)

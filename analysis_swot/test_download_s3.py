import s3fs

# --- Configuration ---
endpoint_url = "https://minio.lab.dive.edito.eu"
bucket = "project-meom-ige"
prefix = "LLC4320"   # folder inside the bucket

# If your bucket is public, use anon=True
# Otherwise, provide your access key and secret key
fs = s3fs.S3FileSystem(
    anon=True,   # or anon=False with credentials below
    # key="YOUR_ACCESS_KEY",
    # secret="YOUR_SECRET_KEY",
    client_kwargs={"endpoint_url": endpoint_url},
)


# List files (non-recursive)
files = fs.ls(f"{bucket}/{prefix}", detail=True)
print(f"Found {len(files)} items under {bucket}/{prefix}\n")


cycle_name = "cycle_025"
files_cycleN = fs.ls(f"{bucket}/{prefix}/{cycle_name}", detail=True)
print(f"Found {len(files_cycleN)} items under {bucket}/{prefix}/{cycle_name}\n")


# Show first few entries with size in MB
for f in files[:10]:
    size_mb = f["Size"] / 1_000_000
    print(f"{f['Key']}  —  {size_mb:.2f} MB")



for f in files_cycleN[:10]:
    size_mb = f["Size"] / 1_000_000
    print(f"{f['Key']}  —  {size_mb:.2f} MB")



# Pick the first file from the list
remote_path = files[0]["Key"]
local_filename = remote_path.split("/")[-1]  # just take basename

print(f"Downloading {remote_path} → {local_filename}")

fs.get(remote_path, local_filename)
print("✅ Download complete!")



# Read it directly (for NetCDF)
import xarray as xr

nc_path = f"s3://{bucket}/{prefix}/somefile.nc"

ds = xr.open_dataset(nc_path, engine="h5netcdf", backend_kwargs={
    "storage_options": {
        "anon": True,
        "client_kwargs": {"endpoint_url": endpoint_url},
    }
})

print(ds)

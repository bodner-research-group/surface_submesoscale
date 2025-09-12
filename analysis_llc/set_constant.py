###### Set constants for all python scripts


# ========== Domain ==========
domain_name = "Station_Papa"
face = 7
i = slice(1873-82,2189+82,1)  # Ocean Weather Station Papa  147W-143W  
j = slice(3408-144,3599+1+145,1) # Ocean Weather Station Papa  48N-52N
# i = slice(1873,2189+1,1)  # Ocean Weather Station Papa  147W-143W  
# j = slice(3408,3599+1,1) # Ocean Weather Station Papa  48N-52N


# # ========== Domain ==========
# domain_name = "SWOT_Site"
# face = 10
# i = slice(2931-113,3186+113,1) # SWOT Validation Site 126W-124W
# j = slice(0,480,1)  # SWOT Validation Site  35N-37N 



# # ========== Domain ==========
# domain_name = "Antarctic_Peninsula"
# face = 12
# i = slice(183-40,586+40,1) 
# j = slice(3168,3743,1)
# ## lat_min, lat_max = -63, -59
# ## lon_min, lon_max = -62, -50 # West longitudes as negative



# # ========== Domain ==========
# domain_name = "icelandic_basin"
# face = 2
# i = slice(527, 1007)   # icelandic_basin -- larger domain
# j = slice(2960, 3441)  # icelandic_basin -- larger domain
# # i=slice(671,864,1)   # icelandic_basin -- small domain, same as Johnson et al. (2016)
# # j=slice(2982,3419,1) # icelandic_basin -- small domain, same as Johnson et al. (2016)

# # ========== Domain ==========
# domain_name = "Southern_Ocean"
# face = 1
# i = slice(600,1080,1) # Southern Ocean
# j = slice(0,481,1)    # Southern Ocean

# # ========== Domain ==========
# domain_name = "Tropics"
# face = 1
# i = slice(520,1000,1) # Tropics
# j = slice(2800,3201,1) # Tropics. --- only 401 grid points, need to change to 481

# i=slice(450,760,1)
# j=slice(450,761,1)


# ========== Time settings ==========
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24







# ################### IF USING LAT/LON TO FIND J, I INDICES
# # Define lat/lon bounds
# lat_min, lat_max = 48, 52
# lon_min, lon_max = -147, -143  # 147W to 143W

# # Create mask
# region_mask = (lat2d >= lat_min) & (lat2d <= lat_max) & \
#               (lon2d >= lon_min) & (lon2d <= lon_max)

# # Find bounding box indices
# j_indices, i_indices = np.where(region_mask)
# j_start, j_end = j_indices.min(), j_indices.max() + 1
# i_start, i_end = i_indices.min(), i_indices.max() + 1

# # Define new slices
# i = slice(i_start, i_end)
# j = slice(j_start, j_end)

# # Grid spacings in m
# dxF = ds1.dxF.isel(face=face,i=i,j=j)
# dyF = ds1.dyF.isel(face=face,i=i,j=j)

# # Coordinate
# lat = ds1.YC.isel(face=face,i=1,j=j)
# lon = ds1.XC.isel(face=face,i=i,j=1)
# depth = ds1.Z

# # Convert lat/lon from xarray to NumPy arrays
# lat_vals = lat.values  # shape (j,)
# lon_vals = lon.values  # shape (i,)

# # Create 2D lat/lon meshgrid
# lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')  # shape (j, i)
# ################### END IF USING LAT/LON TO FIND J, I INDICES


# ### Fix the warning: The input coordinates to pcolormesh are interpreted as cell centers, but are not monotonically increasing or decreasing. This may lead to incorrectly calculated cell edges, in which case, please supply explicit cell edges to pcolormesh.
# def compute_edges(arr):
#     edges = np.zeros(len(arr) + 1)
#     edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
#     edges[0] = arr[0] - (arr[1] - arr[0]) / 2
#     edges[-1] = arr[-1] + (arr[-1] - arr[-2]) / 2
#     return edges

# lon_edges = compute_edges(lon_vals)
# lat_edges = compute_edges(lat_vals)

# lon2d_edges, lat2d_edges = np.meshgrid(lon_edges, lat_edges, indexing='xy')

# lon2d = lon2d_edges
# lat2d = lat2d_edges
# ### End 

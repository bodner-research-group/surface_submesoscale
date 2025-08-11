###### Set constants for all python scripts

k_surf = 0

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

# ========== Domain ==========
domain_name = "Tropics"
face = 1
i = slice(520,1000,1) # Tropics
j = slice(2800,3201,1) # Tropics

# # i=slice(450,760,1)
# # j=slice(450,761,1)

# ========== Time settings ==========
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24

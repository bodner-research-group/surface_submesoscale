# Convert images to video

import os

figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_time_k0"
output_movie = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/movie-face01_time_k0.mp4"

os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/tt_face01_time*_k0.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")
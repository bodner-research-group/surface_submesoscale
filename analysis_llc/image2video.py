##### Convert images to video
import os
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_time_k0"
output_movie = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/movie-face01_time_k0.mp4"
os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/tt_face01_time*_k0.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")


##### Convert images to video
import os
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/strain_vorticity"
# high-resolution
output_movie = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/movie-strain_vorticity-hires.mp4"
os.system(f"ffmpeg -r 10 -pattern_type glob -i '{figdir}/combined_norm_map_*.png' -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")
# low-resolution
output_movie = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/movie-strain_vorticity-lores.mp4"
cmd = (
    f"ffmpeg -y -r 10 -pattern_type glob -i '{figdir}/combined_norm_map_*.png' "
    f"-vf scale=iw/2:ih/2 "
    f"-vcodec mpeg4 "
    f"-q:v 1 "
    f"-pix_fmt yuv420p "
    f"{output_movie}"
)
os.system(cmd)


##### Convert images to video
import os
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/strain_vorticity"
output_movie = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin/movie-jointPDF.mp4"
os.system(f"ffmpeg -r 5 -pattern_type glob -i '{figdir}/joint_pdf_sigma_zeta_week*.png' -vf scale=iw/2:ih/2  -vcodec mpeg4 -q:v 1 -pix_fmt yuv420p {output_movie}")


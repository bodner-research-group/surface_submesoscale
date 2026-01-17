clear;close all;
addpath colormaps bathymetry/
fontsize = 17;
% load_colors;


topo_all = zeros(3600*2,3600*3);
lat_all = zeros(3600*2,1);
lon_all = zeros(3600*3,1);

ncfile = "ETOPO_2022_v1_15s_N60W015_surface.nc";
topo1 = ncread(ncfile,"z");
lat1 = ncread(ncfile,'lat');
lon1 = ncread(ncfile,'lon');
topo_all(1:3600,1:3600) = topo1;
lat_all(1:3600) = lat1;
lon_all(1:3600) = lon1;

ncfile = "ETOPO_2022_v1_15s_N60W030_bed.nc";
topo1 = ncread(ncfile,'z');
lon1 = ncread(ncfile,'lon');
topo_all(1:3600,3600+1:3600*2) = topo1;
lon_all(3600+1:3600*2) = lon1;

ncfile = "ETOPO_2022_v1_15s_N60W045_bed.nc";
topo1 = ncread(ncfile,'z');
lon1 = ncread(ncfile,'lon');
topo_all(1:3600,3600*2+1:3600*3) = topo1;
lon_all(3600*2+1:3600*3) = lon1;



ncfile = "ETOPO_2022_v1_15s_N75W015_bed.nc";
topo1 = ncread(ncfile,"z");
lat1 = ncread(ncfile,'lat');
topo_all(3600+1:3600*2,1:3600) = topo1;
lat_all(3600+1:3600*2) = lat1;

ncfile = "ETOPO_2022_v1_15s_N75W030_bed.nc";
topo1 = ncread(ncfile,'z');
topo_all(3600+1:3600*2,3600+1:3600*2) = topo1;

ncfile = "ETOPO_2022_v1_15s_N75W045_bed.nc";
topo1 = ncread(ncfile,'z');
topo_all(3600+1:3600*2,3600*2+1:3600*3) = topo1;


figure(1)
clf;
pcolor(lon_all,lat_all,topo_all);
shading flat;
colorbar;
set(gca,'fontsize',17)

%% 
%%%%% Figure 1:
%%%%% Panel 1: bathymetry 

clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')


addpath colormaps bathymetry colormaps/slanCM
load_colors;

%%% ===== Load data =====
ncfile = 'ETOPO_2022_v1_60s_N90W180_surface.nc';
lat = ncread(ncfile,'lat');
lon = ncread(ncfile,'lon');
topo = ncread(ncfile,'z');

%%% ===== Crop to your region =====
latlim = [49.99+3 70.01-3];
lonlim = [-41.68+5 0.01-7];

lat_idx = find(lat >= latlim(1) & lat <= latlim(2));
lon_idx = find(lon >= lonlim(1) & lon <= lonlim(2));

lat_sub = lat(lat_idx);
lon_sub = lon(lon_idx);
topo_sub = topo(lon_idx, lat_idx);

%%% ===== Generate meshgrid =====
[Lon, Lat] = meshgrid(lon_sub, lat_sub);

%%% ===== Colormap =====
mycolor2 = flip(cmocean('ice'));
mycolor  = [boxcolor; mycolor2];

fontsize = 18;

%%% ===== Figure and axes =====
lat0 = mean(lat_sub);
lon0 = mean(lon_sub);


%%
figure(1); clf; set(gcf,'Color','w','Position', [584 495 560 420]);
axesm('lambert', ...
      'MapLatLimit', latlim, ...
      'MapLonLimit', lonlim, ...
      'MapParallels', [55 60], ...  
      'Frame', 'on', ...
      'Grid', 'on', ...
      'MeridianLabel','on', ...
      'ParallelLabel','on',...
      'MLineLocation', 4,... % Meridian gridlines every 4 degree
      'PLineLocation', 2);   % Parallel gridlines every 2 degree

%%% ===== Plot bathymetry =====
pcolorm(Lat, Lon, -topo_sub'/1000);
shading flat;

% axis off; 
hold on;

%%% ===== Plot zero contour =====
black = [0 0 0];
contourm(Lat, Lon, -topo_sub'/1000, [0 0], 'EdgeColor', darkgray, 'LineWidth', 0.7);



%%% ===== Plot box =====
lon_box = [-27  -17.5  -17.5  -27  -27];
lat_box = [ 57.9    57.9     62.3     62.3    57.9];
plotm(lat_box, lon_box, 'k--', 'LineWidth', 2);

%%% ===== Colormap & colorbar =====
colormap(mycolor);
clim([0 5]);

cb = colorbar;
cb.Label.String = 'Depth (km)';
cb.Ticks = 0:1:5;
set(cb,'YDir','reverse','Position',[0.79 0.48 0.015 0.4]);

set(gca,'fontsize',fontsize);
box off

figdir = '~/surface_submesoscale/Manuscript_Figures/figure1/';
print(gcf,'-dpng','-r300',[figdir 'fig1_bathymetry.png']);


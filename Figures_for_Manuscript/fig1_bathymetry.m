%%%%% Figure 1:
%%%%% Panel 1: bathymetry 

clear; close all;
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
% mycolor1 = flip(cmocean('diff'));
% mycolor1 = mycolor1(1:2:128,:);
% mycolor2 = flip(cmocean('ice'));
% mycolor2 = mycolor2(1:2:end,:);
% mycolor  = [mycolor1; mycolor2];

mycolor2 = flip(cmocean('ice'));
mycolor  = [boxcolor; mycolor2];

fontsize = 17;

%%% ===== Figure and axes =====
lat0 = mean(lat_sub);
lon0 = mean(lon_sub);


%%
figure(1); clf; set(gcf,'Color','w');

axesm('lambert', ...
      'MapLatLimit', latlim, ...
      'MapLonLimit', lonlim, ...
      'MapParallels', [55 60], ...  
      'Frame', 'on', ...
      'Grid', 'on', ...
      'MeridianLabel','on', ...
      'ParallelLabel','on');

% Adjust grid line spacing manually
setm(gca, 'Grid', 'on'); % Enable grid lines

% Set the spacing for grid lines (in degrees)
setm(gca, 'MLineLocation', 4);  % Meridian gridlines every 1 degree
setm(gca, 'PLineLocation', 2);  % Parallel gridlines every 1 degree

% 
% % Adjust appearance of the gridlines
% setm(gca, 'GridColor', 'k');  % Change grid color to black
% setm(gca, 'GridLineStyle', '--');  % Dashed gridlines

%%% ===== Plot bathymetry =====
pcolorm(Lat, Lon, -topo_sub'/1000);
shading flat;

% axis off; 
hold on;

%%% ===== Plot zero contour =====
black = [0 0 0];
contourm(Lat, Lon, -topo_sub'/1000, [0 0], 'EdgeColor', black, 'LineWidth', 1.2);
% contourm(Lat, Lon, -topo_sub'/1000, [1 1], '-','EdgeColor', black, 'LineWidth', 0.5);
% contourm(Lat, Lon, -topo_sub'/1000, [2 2], '-','EdgeColor', black, 'LineWidth', 0.5);
% contourm(Lat, Lon, -topo_sub'/1000, [3 3], '-','EdgeColor', black, 'LineWidth', 0.5);
% contourm(Lat, Lon, -topo_sub'/1000, [4 4], '-','EdgeColor', black, 'LineWidth', 0.5);

%%% ===== Set grid lines interval using m_grid =====
% Initialize the map projection (if necessary)
% m_proj('lambert', 'lat', latlim, 'lon', lonlim); 

% Use m_grid to control grid intervals manually
% m_grid('parallel', 2, 'meridian', 5, 'LineStyle', '--', 'GridColor', 'k', 'LineWidth', 0.5);


%%% ===== Plot box =====
lon_box = [-27  -17.3  -17.3  -27  -27];
lat_box = [ 58    58     62     62    58];
plotm(lat_box, lon_box, 'k--', 'LineWidth', 2);


%%% ===== Colormap & colorbar =====
colormap(mycolor);
% clim([-length(mycolor1)/length(mycolor2) 1]*5);
clim([0 5]);

cb = colorbar;
cb.Label.String = 'Depth (km)';
cb.Ticks = 0:1:5;
set(cb,'YDir','reverse');

set(gca,'fontsize',fontsize);
box off

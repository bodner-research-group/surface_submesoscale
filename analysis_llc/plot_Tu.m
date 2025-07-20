clear all;close all;
figdir = "figs/face01_day1_3pm";
ncfile = fullfile(figdir, "Tu_difference.nc");

info = ncinfo(ncfile);
vars = {info.Variables.Name};  

for i = 1:length(vars)
    varname = vars{i};
    assignin('base', varname, ncread(ncfile, varname));
end

clear vars info i ncfile varname
%%
% figure(20)
% pcolor(lon2d,lat2d,Tu_abs_diff)
% colorbar;shading flat;

% Create figure
fontsize = 20;

figure(1);clf;set(gcf,'Color','w')
hold on;
box on;
% grid on;
xlabel('\partial S (psu/m)');
ylabel('\partial \theta (^oC/m)');
title('Surface Horizontal Turner Angle');

% % 1. Plot dt_cross vs ds_cross as scatter
% scatter(ds_cross(:), dt_cross(:), 10, 'filled', 'MarkerFaceAlpha', 0.2);

% 2. Overlay constant density lines using linear EOS
S_line = linspace(min(ds_cross(:),[],'omitnan'), max(ds_cross(:),[],'omitnan'), 100);
slope_rho = mean(beta_surf,"all")/ mean(alpha_surf,"all");

min_ds = min(ds_cross(:),[],'omitnan');
max_ds = max(ds_cross(:),[],'omitnan');

min_dt = min(dt_cross(:),[],'omitnan');
max_dt = max(dt_cross(:),[],'omitnan');
c_values = min_dt*2:(max_dt-min_dt)/9:max_dt*2;

for c = c_values  % adjust for a few density anomaly lines
    T_line = slope_rho * S_line + c; % iso-density lines
    plot(S_line, T_line, '-', 'Color', [0.5 0.5 0.5]);
end

ylim([min_dt max_dt])
xlim([min_ds max_ds])



% 3. Custom dotted grid
ax = gca; grid off;
xticks_vals = xticks; yticks_vals = yticks;
xlim_vals = xlim; ylim_vals = ylim;

for x = xticks_vals
    line([x x], ylim_vals, 'Color', [0.7 0.7 0.7], 'LineStyle', ':', 'HandleVisibility','off');
end
for y = yticks_vals
    line(xlim_vals, [y y], 'Color', [0.7 0.7 0.7], 'LineStyle', ':', 'HandleVisibility','off');
end

set(gca,'Fontsize',fontsize)


%%
% 4. Define isopycnal and cross-isopycnal unit vectors
v_cross = [1; -1/slope_rho];  % cross-isopycnal 
v_cross = v_cross / norm(v_cross);
v_iso = [1/slope_rho; 1];     % isopycnal
v_iso = v_iso / norm(v_iso);


% 5. Plot Turner angle PDF as red direction lines
for n = 1:length(x_grid_h)
    angle_deg = x_grid_h(n);         % angle in dgrees
    dir_vec = cosd(angle_deg)*v_cross + sind(angle_deg)*v_iso;  % rotate from cross-isopycnal
   
    mag = pdf_values_h(n) * 0.01;         % scaled length for visibility
    x = [0, mag * dir_vec(1)];
    y = [0, mag * dir_vec(2)];
    plot(x, y, 'r', 'LineWidth', 0.7);

    mag = pdf_values_v(n) * 0.002;         % scaled length for visibility
    x = [0, mag * dir_vec(1)];
    y = [0, mag * dir_vec(2)];
    plot(x, y, 'g', 'LineWidth', 0.7);

end

% Optional legend
% legend({'Cross-isopycnal gradients', 'Constant density lines', 'Turner angle PDF'}, 'Location', 'best');
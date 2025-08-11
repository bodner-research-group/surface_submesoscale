clear all;% close all;
addpath colormap/
load_colors;

% figdir = 'figs/icelandic_basin-summer';
figdir = 'figs/icelandic_basin';
ncfile = fullfile(figdir, 'Tu_difference.nc');

info = ncinfo(ncfile);
vars = {info.Variables.Name};  

for i = 1:length(vars)
    varname = vars{i};
    assignin('base', varname, ncread(ncfile, varname));
end

clear vars info i ncfile varname



%%%%%%%
beta = mean(beta_surf,"all");
alpha = mean(alpha_surf,"all");
ds_cross = ds_cross*beta;
dt_cross = dt_cross*alpha;

%%%%%%%
fontsize = 20;

figure(1);clf;set(gcf,'Color','w','Position',[111 221 759 511])
hold on;
box on;
xlabel('\beta\partial S ');
ylabel('\alpha\partial \theta ');
title('Kernel PDF of Turner Angle');

S_line = linspace(min(ds_cross(:),[],'omitnan')*7, max(ds_cross(:),[],'omitnan')*7, 400);
% slope_rho = mean(beta_surf,"all")/ mean(alpha_surf,"all");
slope_rho = 1;
min_ds = min(ds_cross(:),[],'omitnan');
max_ds = max(ds_cross(:),[],'omitnan');

min_dt = min(dt_cross(:),[],'omitnan');
max_dt = max(dt_cross(:),[],'omitnan');
c_values = linspace(min_dt*10, max_dt*10, 100);  

for c = c_values  % adjust for a few density anomaly lines
    T_line = slope_rho * S_line + c; % iso-density lines
    plot(S_line, T_line, '-', 'Color', [0.5 0.5 0.5]);
end

set(gca,'Fontsize',fontsize)


%%
% 4. Define isopycnal and cross-isopycnal unit vectors
v_cross = [-slope_rho; 1];  % cross-isopycnal 
v_iso = [1;slope_rho];     % isopycnal


v_cross = v_cross / norm(v_cross);
v_iso = v_iso / norm(v_iso);

x_grid = x_grid_h;

% 5. Plot Turner angle PDF as direction lines
h_pdf_hori = [];  % for legend
h_pdf_vert = [];

x_all = []; y_all = [];
scale = 7e-8;   


for n = 1:length(x_grid)
    angle_deg = x_grid(n);
    dir_vec = cosd(angle_deg)*v_cross + sind(angle_deg)*v_iso;

    % Horizontal Turner angle PDF
    mag = pdf_values_h(n) * scale*3;
    x = [0, mag * dir_vec(1)];
    y = [0, mag * dir_vec(2)];
    h = plot(x, y, 'Color',blue, 'LineWidth', 0.7);
    if isempty(h_pdf_hori)
        h_pdf_hori = h;
    end
    x_all(end+1) = x(2);  % collect endpoint
    y_all(end+1) = y(2);

    % Vertical Turner angle PDF
    mag = pdf_values_v(n) * scale;
    x = [0, mag * dir_vec(1)];
    y = [0, mag * dir_vec(2)];
    h = plot(x, y, 'Color',green, 'LineWidth', 0.7);
    if isempty(h_pdf_vert)
        h_pdf_vert = h;
    end
    x_all(end+1) = x(2);
    y_all(end+1) = y(2);
end

padding = 0.02;  % add 2% margin
x_min = min(x_all); x_max = max(x_all);
y_min = min(y_all); y_max = max(y_all);

x_range = x_max - x_min;
y_range = y_max - y_min;

xlim([x_min - padding*x_range, x_max + padding*x_range]);
ylim([y_min - padding*y_range, y_max + padding*y_range]);

% Add horizontal and vertical zero lines
plot(get(gca, 'XLim'), [0 0], 'k--', 'LineWidth', 1);  % horizontal zero line
plot([0 0], get(gca, 'YLim'), 'k--', 'LineWidth', 1);  % vertical zero line


% Optional legend (cleaner)
legend([h_pdf_hori, h_pdf_vert], ...
       {'Horizontal Turner angle PDF', 'Vertical Turner angle PDF'}, ...
       'Location', 'northeast', 'FontSize', fontsize - 4);


% ========== plot cross-isopycnal vector and isopycnal vector ==========
origin = [0, 0]; 

% cross-isopycnal vector
quiver(origin(1), origin(2), v_cross(1)*scale/50, v_cross(2)*scale/50, 0, ...
    'r', 'LineWidth', 2, 'MaxHeadSize', 0.4, 'DisplayName', '⊥ isopycnal');

% isopycnal vector
quiver(origin(1), origin(2), v_iso(1)*scale/50, v_iso(2)*scale/50, 0, ...
    'b', 'LineWidth', 2, 'MaxHeadSize', 0.4, 'DisplayName', '∥ isopycnal');

legend('show')

% print('-dpng','-r200',[figdir '/Tu_TS_newAxes.png']);

% % axis equal

% print('-dpng','-r200',[figdir '/Tu_TS-equal.png']);

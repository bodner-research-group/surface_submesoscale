
clear;close all;

set(groot, 'DefaultFigureRenderer', 'painters')

load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/timeseries_surface_forcing.mat')

% Convert time strings to datetime
time_B0 = datetime(time_B0, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_wind = datetime(time_wind, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_Hml = datetime(time_Hml, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');

% Define start and end of time
t_start = dateshift(min(time_B0), 'start', 'month');
t_end   = dateshift(max(time_B0), 'start', 'month') + calmonths(1);

% Generate monthly ticks (every month)
monthly_ticks = t_start : calmonths(1) : t_end;

load_colors;

POSITION = [52 526 764 291];
fontsize = 16;
figdir = '~/surface_submesoscale/Manuscript_Figures/figure1/';

figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [52 526 764+60 291])
set(gca, 'Position', [0.1 0.1100 0.725 0.8150])

% Add 12 grid lines per year (1 per month)
for t = monthly_ticks
    xline(t, 'Color', [0.8 0.8 0.8], 'LineStyle', ':', 'LineWidth', 0.5); % light gray lines
end

hold on;

% Left y-axis:
yyaxis left
l11 = plot(time_B0, Bflux_daily_avg*1e7, 'Color',blue, 'LineWidth', 2);
ylabel('\boldmath$B_0\ (\mathrm{m^2/s^3})$','Interpreter','latex');
ax1 = gca;
ax1.YColor =  blue;
set(gca,'FontSize', fontsize);
l111 = plot(time_B0, zeros(1,length(time_B0)),'--' ,'Color',blue, 'LineWidth', 0.75);

% Right y-axis: H_ml
yyaxis right
l12 = plot(time_wind, tau_mag_mean,'k-.', 'LineWidth', 2.5);
% ylabel('\boldmath$H_\mathrm{ML} (\mathrm{m})$','Interpreter','latex');
ax1.YColor = 'k';  % Black axis
set(gca,'FontSize', fontsize);
           

annotation('textbox', [0.1, 0.82, 0.3, 0.1], ...   % [x, y, width, height] in normalized figure units
           'String', 'Surface buoyancy flux', ...
           'Color', blue, ...               
           'FontSize', fontsize+1, ...
           'EdgeColor', 'none');    

annotation('textbox', [0.63, 0.77, 0.2, 0.1], ...   % [x, y, width, height] in normalized figure units
           'String', 'Surface wind stress', ...
           'Color', 'k', ...               
           'FontSize', fontsize+1, ...
           'EdgeColor', 'none');    

% Define custom tick locations (every 2 months from Nov 2011 to Nov 2012)
xticks_custom = datetime(2011,11,1) : calmonths(2) : datetime(2012,11,1);
xlim([datetime(2011,11,1), datetime(2012,11,1)]);
xticks(xticks_custom);
xticklabels(datestr(xticks_custom, 'mmm yyyy'));  % or use datestr for full control

grid(ax1, 'on'); grid(ax1, 'minor');
set(gca, 'FontSize', fontsize);

box(ax1, 'on');       % For the main axes

% print(gcf,'-dpng','-r300',[figdir 'fig1_timeseries1.png']);


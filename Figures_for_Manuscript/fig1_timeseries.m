
clear;close all;

% set(groot, 'DefaultFigureRenderer', 'painters')

load('~/surface_submesoscale/figs/icelandic_basin/timeseries_data_new.mat')
load_colors

POSITION = [52 526 764 291];
fontsize = 16;
figdir = '~/surface_submesoscale/Figures_for_Manuscript/figure1/';

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
l11 = plot(time_Hml, Bflux_daily_avg*1e7, 'Color',blue, 'LineWidth', 2);
ylabel('\boldmath$B_0\ (\mathrm{m^2/s^3})$','Interpreter','latex');
ax1 = gca;
ax1.YColor =  blue;
set(gca,'FontSize', fontsize);
l111 = plot(time_Hml, zeros(1,length(time_Hml)),'--' ,'Color',blue, 'LineWidth', 0.75);

% Right y-axis: H_ml
yyaxis right
l12 = plot(time_Hml, (Hml_mean),'k-.', 'LineWidth', 2.5);
ylabel('\boldmath$H_\mathrm{ML} (\mathrm{m})$','Interpreter','latex');
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


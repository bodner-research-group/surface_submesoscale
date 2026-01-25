%%% Figure 1:

clear; close all;


% set(groot, 'DefaultFigureRenderer', 'painters')

% % ===== Lock graphics defaults (2024a-like) =====
% set(groot, ...
%     'DefaultFigureRenderer', 'painters', ...
%     'DefaultFigureColor', 'w', ...
%     'DefaultAxesFontName', 'Helvetica', ...
%     'DefaultTextFontName', 'Helvetica', ...
%     'DefaultAxesFontSize', 16, ...
%     'DefaultTextFontSize', 16, ...
%     'DefaultLineLineWidth', 1, ...
%     'DefaultAxesTickDir', 'out', ...
%     'DefaultAxesLayer', 'top', ...
%     'DefaultAxesBox', 'on')



load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/timeseries_data_new.mat')

Lambda_MLI_mean = [NaN NaN NaN Lambda_MLI_mean NaN NaN NaN];
N2ml_mean = [NaN NaN NaN N2ml_mean NaN NaN NaN];

addpath ~/surface_submesoscale/analysis_llc/colormap
load_colors


POSITION = [52 526 764 291];
fontsize = 19;
figdir = '~/surface_submesoscale/Manuscript_Figures/figure1/';

% Convert time strings to datetime
time_N2 = datetime(time_N2, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_Hml = datetime(time_Hml, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_Tu = datetime(time_Tu, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_wind = datetime(time_wind, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');

% Handle zeros or negative values in N2ml_mean (replace with small positive value)
N2ml_mean_logsafe = squeeze(N2ml_mean);
N2ml_mean_logsafe(N2ml_mean_logsafe <= 0) = NaN; % or a small positive number like 1e-10 if you prefer

% Define start and end of time
t_start = dateshift(min(time_N2), 'start', 'month');
t_end   = dateshift(max(time_N2), 'start', 'month') + calmonths(1);

% Generate monthly ticks (every month)
monthly_ticks = t_start : calmonths(1) : t_end;
% 
% 
% %%
% 
% figure(10)
% clf;
% plot(time_qnet, mean_spec_in_mld_submeso, 'LineWidth', 2);
% hold on;
% plot(time_qnet, vertical_bf_submeso_mld/4, 'LineWidth', 2);
% plot(time_Hml, -Hml_mean/1e13, 'LineWidth', 2);
% plot(time_qnet, qnet_7day_smooth/0.4e13, 'LineWidth', 2);
% 
% hold off



%%% === Figure 1: (3 y-axes) ===
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [52 526 764+60 291])
set(gca, 'Position', [0.1 0.1100 0.725 0.8150])

% Add 12 grid lines per year (1 per month)
for t = monthly_ticks
    xline(t, 'Color', [0.8 0.8 0.8], 'LineStyle', ':', 'LineWidth', 0.5); % light gray lines
end

hold on;


% Left y-axis: Bflux_7day_smooth
yyaxis left
l11 = plot(time_Hml, Bflux_daily_avg*1e7, 'Color',blue, 'LineWidth', 2);
% l111 = plot(time_qnet, vertical_bf_submeso_mld*300*1e7, 'Color',purple, 'LineWidth', 2);


ylabel('\boldmath$B_0\ (\mathrm{m^2/s^3})$','Interpreter','latex','FontSize',fontsize+1);
% ylabel('\boldmath$\mathrm{Buoyancy\ flux}\ (10^{-7}\,\mathrm{m^2/s^3})$','Interpreter','latex');
ax1 = gca;
% ax1.YColor =  (blue + purple)/2; % Blue axis
ax1.YColor =  blue;
set(gca,'FontSize', fontsize-2);
% ylim([-400 170]);  % Set y-axis limits correctly
l111 = plot(time_Hml, zeros(1,length(time_Hml)),'--' ,'Color',blue, 'LineWidth', 0.75);


% Right y-axis: H_ml
yyaxis right
l12 = plot(time_Hml, -(Hml_mean),'k-.', 'LineWidth', 2.5);
% ylabel('\boldmath$H_\mathrm{ML} (\mathrm{m})$','Interpreter','latex');
ylabel('\boldmath$-\langle H\rangle (\mathrm{m})$','Interpreter','latex');
ax1.YColor = 'k';  % Black axis
set(gca,'FontSize', fontsize-2);
ylim([-800 1])

% Add third y-axis manually (for N^2, log scale)
ax1 = gca;  % current axes
ax1_pos = ax1.Position;  % position of first axes

% annotation('textbox', [0.54, 0.56, 0.25, 0.1], ...   % [x, y, width, height] in normalized figure units
annotation('textbox', [0.51, 0.49, 0.25, 0.1], ...   % [x, y, width, height] in normalized figure units
            'String', 'Mixed layer stratification', ...
           'Color', green, ...               % green color
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');                 % remove box border

annotation('textbox', [0.625, 0.75, 0.25, 0.1], ...   % [x, y, width, height] in normalized figure units
...% annotation('textbox', [0.505, 0.805, 0.25, 0.1], ...   % [x, y, width, height] in normalized figure units
            'String', 'Mixed layer base', ...
           'Color', 'k', ...               
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');               

annotation('textbox', [0.1, 0.83, 0.3, 0.1], ...   % [x, y, width, height] in normalized figure units
           'String', 'Surface buoyancy flux', ...
           'Color', blue, ...               
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');               

% annotation('textbox', [0.23, 0.74, 0.3, 0.1], ...   % [x, y, width, height] in normalized figure units
%            'String', 'Submesoscale vertical buoyancy flux \times300', ...
%            'Color', purple, ...               
%            'FontSize', fontsize+1, ...
%            'EdgeColor', 'none'); 

% Define custom tick locations (every 2 months from Nov 2011 to Nov 2012)
xticks_custom = datetime(2011,11,1) : calmonths(2) : datetime(2012,11,1);
xlim([datetime(2011,11,1), datetime(2012,11,1)]);
xticks(xticks_custom);
xticklabels(datestr(xticks_custom, 'mmm yyyy'));  % or use datestr for full control

% Create ax3 for plotting N2ml_mean (aligned with main axes)
ax3 = axes('Position', ax1_pos, ...
           'Color', 'none', ...
           'YAxisLocation', 'right', ...
           'XAxisLocation', 'bottom', ...
           'XColor', 'none', 'YColor', 'g');
hold(ax3, 'on');
set(ax3, 'YScale', 'log');

plot(ax3, time_Hml, N2ml_mean_logsafe, 'Color',green, 'LineWidth', 2);
ylabel(ax3, '\boldmath$N^2\ (\mathrm{s^{-2}})$','Interpreter','latex','FontSize',fontsize+1);
set(ax3, 'FontSize', fontsize-2);
ylim([3e-7 3e-4])

ax3.YColor = 'none';
ax3.YTick = [1e-6 1e-5 1e-4];

% Now create a new axes only for the y-axis labels and ticks, shifted right
ax3_label = axes('Position', ax1_pos, ...
                 'Color', 'none', ...
                 'YAxisLocation', 'right', ...
                 'XAxisLocation', 'bottom', ...
                 'XColor', 'none', 'YColor', green);

% Shift this axis right
ax3_label.Position(1) = ax3_label.Position(1) + 0.09;

% Hide the plot in ax3_label, only show y-axis
set(ax3_label, 'XTick', [], 'YTick', get(ax3, 'YTick'));
set(ax3_label, 'YLim', get(ax3, 'YLim'), 'YScale', 'log');
ylabel(ax3_label, '\boldmath$N^2\ (\mathrm{s^{-2}})$  (50\%--90\% ML)','Interpreter','latex','FontSize',fontsize+1);
set(ax3_label, 'FontSize', fontsize);

% Hide x-axis and box for the label axis
ax3_label.XColor = 'none';
ax3_label.Box = 'off';
ax3_label.Visible = 'on';

% Link y-limits of both ax3 and ax3_label
linkaxes([ax3, ax3_label], 'y');


grid(ax1, 'on'); grid(ax1, 'minor');

% Legends
% l1112 = legend(ax1, [l11 l12] ,{'Net surface buoyancy flux', 'Mixed layer base'}, ...
%     'Location', 'northwest','FontSize',fontsize+1,'Position',[0.1098+0.04 0.7339+0.025 0.2518 0.1632]);
% legend(ax1, 'boxoff');
% l13 = legend(ax3, {'Mixed layer stratification'},...
%     'TextColor', green, 'Location', 'northeast','FontSize',fontsize+1,...
%     'Position',[0.5273-0.01 0.8095+0.025-0.6 0.2882 0.0876]);
% l1112.Box = 'off';  % Hide box
% l13.Box = 'off';  % Hide box
% legend(ax3, 'boxoff');

set(gca, 'FontSize', fontsize-2);

box(ax1, 'on');       % For the main axes

print(gcf,'-dpng','-r300',[figdir 'fig1_timeseries1.png']);




%%% Figure 1:

clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

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
% time_Tu = datetime(time_Tu, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
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


%%%%%%%%%%%%%% Add Turner angle agreement
fname = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/TurnerAngle_BuoyancyRatio_daily_Timeseries.nc';
Tu_agreements_new_pct = ncread(fname,'Tu_agreements_new_pct');
Tu_diff_means_new = ncread(fname,'Tu_diff_means_new');

date = ncread(fname,'date');
ref_time = datetime(2011,11,1,0,0,0);   % reference time
time_Tu  = ref_time + days(date);       % convert to datetime



%%% === Figure 1: (3 y-axes) ===
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [52 526 764+60+40 330])
set(gca, 'Position', [0.13 0.1100 0.69 0.8150])

% Add 12 grid lines per year (1 per month)
for t = monthly_ticks
    xline(t, 'Color', [0.8 0.8 0.8], 'LineStyle', ':', 'LineWidth', 0.5); % light gray lines
end

hold on;








%%%%%%%%%%%%%% Add Turner angle agreement
yyaxis left
l11 = plot(time_Tu, Tu_agreements_new_pct, 'Color',blue, 'LineWidth', 2);
ylabel('\boldmath$-\langle B_0\rangle\ (\mathrm{m^2/s^3})$','Interpreter','latex','FontSize',fontsize+1);
ax1 = gca;
ax1.YColor =  blue;
ax1.YDir = 'reverse';
set(gca,'FontSize', fontsize-2);

% ylabel(ax1, '\boldmath\%','Interpreter','latex','FontSize',fontsize+1);
ylabel(ax1, '\boldmath{$\%$}','Interpreter','latex','FontSize',fontsize+1);


% Right y-axis: H_ml
yyaxis right
l12 = plot(time_Hml, -(Hml_mean),'k-.', 'LineWidth', 2.5);
ylabel('\boldmath$-\langle H\rangle (\mathrm{m})$','Interpreter','latex');
ax1.YColor = 'k'; 
set(gca,'FontSize', fontsize-2);
ylim([-800 1])

% Add third y-axis manually (for N^2, log scale)
ax1 = gca;  % current axes
ax1_pos = ax1.Position;  % position of first axes

annotation('textbox', [0.51-0.045, 0.235, 0.25, 0.1], ...   % [x, y, width, height] in normalized figure units
            'String', 'Mixed layer stratification', ...
           'Color', green, ...               % green color
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');                 % remove box border

annotation('textbox', [0.625-0.09, 0.75+0.06, 0.25, 0.1], ...   % [x, y, width, height] in normalized figure units
            'String', 'Mixed layer base', ...
           'Color', 'k', ...               
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');               

annotation('textbox', [0.2, 0.7, 0.25, 0.08], ...   % [x, y, width, height] in normalized figure units
           'String', 'Turner angle agreement', ...
           'Color', blue, ...               
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');     

annotation('textbox', [0.642, 0.235, 0.2, 0.1], ...   % [x, y, width, height] in normalized figure units
            'String', 'Surface buoyancy input', ...
           'Color', darkgray, ...               
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
% ylabel(ax3, '\boldmath$\langle N^2\rangle\ (\mathrm{s^{-2}})$','Interpreter','latex','FontSize',fontsize+1);
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
ylabel(ax3_label, '\boldmath$\langle N^2\rangle\ (\mathrm{s^{-2}})$  (50\%--90\% ML)','Interpreter','latex','FontSize',fontsize+1);
set(ax3_label, 'FontSize', fontsize);

% Hide x-axis and box for the label axis
ax3_label.XColor = 'none';
ax3_label.Box = 'off';
ax3_label.Visible = 'on';

% Link y-limits of both ax3 and ax3_label
linkaxes([ax3, ax3_label], 'y');






%%%% Bflux_7day_smooth
ax4 = axes('Position', ax1_pos, ...
           'Color', 'none', ...
           'YAxisLocation', 'right', ...
           'XAxisLocation', 'bottom', ...
           'XColor', 'none', 'YColor', 'g');
hold(ax4, 'on');

lbuoyancy = plot(ax4, time_Hml, -Bflux_daily_avg*1e7, 'Color',black, 'LineWidth', 1); lbuoyancy.Color(4) = 0.5;
lbuoyancy2 = plot(ax4,time_Hml, zeros(1,length(time_Hml)),'--' ,'Color',black, 'LineWidth', 0.5); lbuoyancy2.Color(4) = 0.5;

set(ax4, 'FontSize', fontsize-2);
ylim([-2 2])

ax4.YColor = 'none';
% ax4.YTick = [1e-6 1e-5 1e-4];

% Now create a new axes only for the y-axis labels and ticks, shifted right
ax4_label = axes('Position', ax1_pos, ...
                 'Color', 'none', ...
                 'YAxisLocation', 'left', ...
                 'XAxisLocation', 'bottom', ...
                 'XColor', 'none', 'YColor', darkgray);

% Shift this axis left
ax4_label.Position(1) = ax4_label.Position(1) -0.068;

% Hide the plot in ax3_label, only show y-axis
set(ax4_label, 'XTick', [], 'YTick', get(ax4, 'YTick'));
set(ax4_label, 'YLim', get(ax4, 'YLim'));
ylabel(ax4_label,'\boldmath$-\langle B_0\rangle\ (\mathrm{m^2/s^3})$','Interpreter','latex','FontSize',fontsize+1);
set(ax4_label, 'FontSize', fontsize);

% Hide x-axis and box for the label axis
ax4_label.XColor = 'none';
ax4_label.Box = 'off';
ax4_label.Visible = 'on';

% Link y-limits of both ax3 and ax3_label
linkaxes([ax4, ax4_label], 'y');













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

print(gcf,'-dpng','-r300',[figdir 'fig1_timeseries_with_TurnerAngle.png']);


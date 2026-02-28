%%% Figure 2:

clear;close all;
set(groot, 'DefaultFigureRenderer', 'painters')

load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/timeseries_steric_submesoSSH_GSW0.8.mat')


addpath ~/surface_submesoscale/analysis_llc/colormap
load_colors

POSITION = [52 526 764 291];
fontsize = 19;
figdir = '~/surface_submesoscale/Manuscript_Figures/figure2/';

% Convert time strings to datetime
time_SSH = datetime(time_SSH, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_SSH_mean = datetime(time_SSH_mean, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_steric = datetime(time_steric, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_submesoSSH = datetime(time_submesoSSH, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_lambda = datetime(time_lambda, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');


% Define start and end of time
t_start = dateshift(min(time_submesoSSH), 'start', 'month');
t_end   = dateshift(max(time_submesoSSH), 'start', 'month') + calmonths(1);

% Generate monthly ticks (every month)
monthly_ticks = t_start : calmonths(1) : t_end;

%%% === Figure 1:  ===
figure(1); clf;
% set(gcf, 'Color', 'w', 'Position', [52 526 764+60 291])
set(gcf, 'Color', 'w', 'Position', [52 526 1000 350])
set(gca, 'Position', [0.1 0.1100 0.725 0.8150])

% Add 12 grid lines per year (1 per month)
for t = monthly_ticks
    xline(t, 'Color', [0.8 0.8 0.8], 'LineStyle', ':', 'LineWidth', 0.5); % light gray lines
end

hold on;





%%
% Left y-axis: Gradient magnitude of mixed-layer steric height
yyaxis left
l11 = plot(time_steric, 1e6*eta_steric_grad_mean, 'Color',brown, 'LineWidth', 2);

ylabel('\boldmath$\langle\vert\nabla \eta_\mathrm{steric}\vert\rangle\ (10^{-6}\,\mathrm{m})$','Interpreter','latex');
ax1 = gca;
ax1.YColor =  brown;
set(gca,'FontSize', fontsize);
ylim([0 1.2])
% l111 = plot(time_steric, zeros(1,length(time_steric)),'--' ,'Color',brown, 'LineWidth', 0.75);




%%
% Right y-axis: Gradient magnitude of submesoscale SSH
yyaxis right
l12 = plot(time_submesoSSH, 1e6*eta_submeso_grad_mean,'Color',olive, 'LineWidth', 2);
% ylabel('\boldmath$H_\mathrm{ML} (\mathrm{m})$','Interpreter','latex');
ylabel('\boldmath$\langle\vert\nabla\eta_\mathrm{submeso}\vert\rangle\ (10^{-6}\,\mathrm{m})$','Interpreter','latex');
ax1.YColor = olive;  % Black axis
set(gca,'FontSize', fontsize);
ylim([0 1.2])


%%
% Add third y-axis manually 
ax1 = gca;  % current axes
ax1_pos = ax1.Position;  % position of first axes

annotation('textbox', [0.4928, 0.52, 0.25, 0.1], ...   % [x, y, width, height] in normalized figure units
           'String', 'Wavelength of the most unstable mode', ...
           'Color', black, ...               % olive color
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');                 % remove box border

annotation('textbox', [0.38, 0.79, 0.3, 0.1], ...   % [x, y, width, height] in normalized figure units
           'String', 'Submesoscale SSH', ...
           'Color', olive, ...               
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');               

annotation('textbox', [0.22, 0.52, 0.15, 0.1], ...   % [x, y, width, height] in normalized figure units
           'String', 'Mixed-layer steric height', ...
           'Color', brown, ...               
           'FontSize', fontsize+2, ...
           'EdgeColor', 'none');               

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
           'XColor', 'none', 'YColor', 'k');
hold(ax3, 'on');
% set(ax3, 'YScale', 'log');

plot(ax3, time_lambda, Lambda_MLI_mean, '-.','Color',black, 'LineWidth', 2);
set(ax3, 'FontSize', fontsize);
% ylim([0 24])
ylim([0 30])

ax3.YColor = 'none';
% ax3.YTick = [0:4:24];
ax3.YTick = [0:5:30];

% Now create a new axes only for the y-axis labels and ticks, shifted right
ax3_label = axes('Position', ax1_pos, ...
                 'Color', 'none', ...
                 'YAxisLocation', 'right', ...
                 'XAxisLocation', 'bottom', ...
                 'XColor', 'none', 'YColor', black);

% Shift this axis right
ax3_label.Position(1) = ax3_label.Position(1) + 0.09;

% Hide the plot in ax3_label, only show y-axis
set(ax3_label, 'XTick', [], 'YTick', get(ax3, 'YTick'));
set(ax3_label, 'YLim', get(ax3, 'YLim'));
% set(ax3_label, 'YLim', get(ax3, 'YLim'), 'YScale', 'log');
ylabel(ax3_label, '\boldmath$\langle\lambda_\mathrm{MLI}\rangle\ (\mathrm{km})$ ','Interpreter','latex');
set(ax3_label, 'FontSize', fontsize);

% Hide x-axis and box for the label axis
ax3_label.XColor = 'none';
ax3_label.Box = 'off';
ax3_label.Visible = 'on';

% Link y-limits of both ax3 and ax3_label
linkaxes([ax3, ax3_label], 'y');


grid(ax1, 'on'); grid(ax1, 'minor');

set(gca, 'FontSize', fontsize);

box(ax1, 'on');       % For the main axes

print(gcf,'-dpng','-r300',[figdir 'fig2_timeseries.png']);




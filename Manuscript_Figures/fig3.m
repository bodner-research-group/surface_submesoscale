%%%% Figure 3 — Individual lines with legend

clear; close all;

% Load data
load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/Hml_tendency_all_plot_data.mat')

fontsize = 19;
figdir = '~/surface_submesoscale/Manuscript_Figures/figure3/';
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [52 526 1000 1000])

%% ============================================================
%% Panel (a): Hml tendency (raw + 7-day rolling)
%% ============================================================

subplot(2,1,1)
hold on

t  = datetime(time , 'ConvertFrom','datenum');     % raw time
tr = datetime(time_rm , 'ConvertFrom','datenum'); % rolling mean time
t_wbeddy  = datetime(time_wbeddy , 'ConvertFrom','datenum');     % raw time
tr_wbeddy  = datetime(time_wbeddy_rm , 'ConvertFrom','datenum');     % raw time

% ----- semi-transparent raw lines -----
p1 = plot(t, dHml_dt, 'Color','k');     p1.Color(4) = 0.25;
p2 = plot(t, vert, 'Color','b');        p2.Color(4) = 0.25;
p3 = plot(t, tendency_ekman, 'Color','c'); p3.Color(4) = 0.25;
p4 = plot(t, diff, 'Color',[0 0.5 0]);   p4.Color(4) = 0.25;
p5 = plot(t, hori_steric, 'Color','m');  p5.Color(4) = 0.25;
p6 = plot(t, hori_submeso, 'Color','r'); p6.Color(4) = 0.25;
p7 = plot(t_wbeddy, wb_eddy, 'Color',[0.5 0 0.5]); p7.Color(4) = 0.25;
p8 = plot(t, residual, 'Color',[0.5 0.5 0.5]); p8.Color(4) = 0.25;

% ----- 7-day rolling mean (solid, bold) -----
p1r = plot(tr, dHml_dt_rm, 'k', 'LineWidth', 2);
p2r = plot(tr, vert_rm, 'b', 'LineWidth', 2);
p3r = plot(tr, tendency_ekman_rm, 'c', 'LineWidth', 2);
p4r = plot(tr, diff_rm, 'Color',[0 0.5 0], 'LineWidth', 2);
p5r = plot(tr, hori_steric_rm, 'Color','m', 'LineWidth', 2);
p6r = plot(tr, hori_submeso_rm, 'Color','r', 'LineWidth', 2);
p7r = plot(tr_wbeddy, wb_eddy_rm, 'Color',[0.5 0 0.5], 'LineWidth', 2);
p8r = plot(tr, residual_rm,'Color', [0.5 0.5 0.5], 'LineWidth', 2);

grid on; grid minor; box on;
ylabel('dH_{ml}/dt  [m/day]')
% xlabel('Time')
title('Mixed Layer Depth Tendency')
set(gca,'FontSize', fontsize)

legend([p1r p2r p3r p4r p5r p6r p7r],...
    {'dHml/dt','Surface buoyancy flux','Ekman buoyancy flux',...
        'dHml-diff','Steric','SSH submeso','B_{eddy}','Residual'}, ...
        'FontSize', fontsize)
legend('boxoff')

ylim([-50 50])


%% ============================================================
%% Panel (b): Reconstructed winter MLD
%% ============================================================

subplot(2,1,2)
hold on

tw = datetime(time_rec,'ConvertFrom','datenum');
tw_steric = datetime(time_Hml_recon_steric_2012,'ConvertFrom','datenum');
% tw_wbeddy = datetime(time_Hml_recon_wbeddy_2012,'ConvertFrom','datenum');

h1 = plot(tw, Hml_mean_2012, 'Color','k', 'LineWidth', 2);
h2 = plot(tw_steric, Hml_recon_steric_2012, 'Color',[0.82 0 0.49], 'LineWidth', 1.8);
h3 = plot(tw_steric, Hml_recon_submeso_2012, 'Color',[1 0.55 0], 'LineWidth', 1.8);
% h4 = plot(tw_wbeddy, Hml_recon_wbeddy_2012, 'Color',[0.5 0 0.5], 'LineWidth', 1.8);

grid on; grid minor;box on;
ylabel('H_{ml}  [m]')
% xlabel('Time')
title('Reconstructed MLD (winter)')
set(gca,'FontSize', fontsize)

% legend({'Total MLD','Reconstructed steric','Reconstructed SSH submeso','Reconstructed B_{eddy}'}, ...
%        'Location','eastoutside', 'FontSize', 10)

legend({'Total MLD','Reconstructed steric',...
    'Reconstructed SSH submeso'}, ...
        'FontSize', fontsize)
legend('boxoff')

xlim([tw(38) tw(214)])
ylim([0 800])

%% ============================================================
%% Save figure
%% ============================================================

if ~exist(figdir,'dir')
    mkdir(figdir)
end

% print(gcf, [figdir 'figure3_timeseries'], '-dpng', '-r300')

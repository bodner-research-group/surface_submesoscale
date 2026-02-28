%%%% Figure 3 — panels (a) and (b)

clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

% Load data
load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/Hml_tendency_all_plot_data_GSW0.8.mat')

addpath ~/surface_submesoscale/analysis_llc/colormap
load_colors

fontsize = 19;
figdir = '~/surface_submesoscale/Manuscript_Figures/figure3/';
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [52 526 800 800])

%% ============================================================
%%% Panel (a): Hml tendency (raw + 7-day rolling)
%%% ============================================================

ax1 = axes('Position',[0.09 0.52 0.85 0.44]);
hold(ax1,'on')

t  = datetime(time , 'ConvertFrom','datenum');     % raw time
tr = datetime(time_rm , 'ConvertFrom','datenum'); % rolling mean time
t_wbeddy  = datetime(time_wbeddy , 'ConvertFrom','datenum');     % raw time
tr_wbeddy  = datetime(time_wbeddy_rm , 'ConvertFrom','datenum');     % raw time

% ----- semi-transparent raw lines -----
p2 = plot(t, vert, 'Color',blue);        p2.Color(4) = 0.25;
p3 = plot(t, tendency_ekman, 'Color',cyan); p3.Color(4) = 0.25;
p4 = plot(t, diff, 'Color',lightred2);   p4.Color(4) = 0.25;
p5 = plot(t, hori_steric, 'Color',brown);  p5.Color(4) = 0.25;
p6 = plot(t, hori_submeso, 'Color',olive); p6.Color(4) = 0.25;
p7 = plot(t_wbeddy, wb_eddy, '-.','Color',purple2); p7.Color(4) = 0.25;
% p8 = plot(t, residual, 'Color',lightgray); p8.Color(4) = 0.25;
p1 = plot(t, dHml_dt,'-','Color',black);     p1.Color(4) = 0.25;

% ----- 7-day rolling mean (solid, bold) -----
p2r = plot(tr, vert_rm, 'Color',blue, 'LineWidth', 2);
p3r = plot(tr, tendency_ekman_rm, 'Color',cyan, 'LineWidth', 2);
p4r = plot(tr, diff_rm, 'Color',lightred2, 'LineWidth', 2); p4r.Color(4) = 0.6;
p5r = plot(tr, hori_steric_rm,'-.', 'Color',brown, 'LineWidth', 2);
p6r = plot(tr, hori_submeso_rm,'-', 'Color',olive, 'LineWidth', 2);
p7r = plot(tr_wbeddy, wb_eddy_rm, '-.','Color',purple2, 'LineWidth', 2);
% p8r = plot(tr, residual_rm,'Color', gray, 'LineWidth', 2.5);
p1r = plot(tr, dHml_dt_rm,'-','Color',black, 'LineWidth', 1.5);

grid on; grid minor; box on;
ylabel('\textbf{$\partial_t\langle H\rangle$  (m/day)}','Interpreter','latex')
% xlabel('Time')
set(gca,'FontSize', fontsize)

title('Mixed layer depth tendency','FontWeight','normal',...
    'FontSize', fontsize+5)

ylim([-60 60])
yticks([-45:15:45])

xt = datetime([2011 11 1; 
               2012  1 1;
               2012  3 1;
               2012  5 1;
               2012  7 1;
               2012  9 1;
               2012 11 1]);

xticks(xt)
xtickformat('MMM yyyy')

xlim([t(1) datetime(2012,11,1)])


% ---- Create transparent overlay axes ----
ax1b = axes('Position',ax1.Position,'Color','none', 'XTick',[], 'YTick',[],'HitTest','off');
ax1c = axes('Position',ax1.Position,'Color','none', 'XTick',[], 'YTick',[],'HitTest','off');
ax1d = axes('Position',ax1.Position,'Color','none', 'XTick',[], 'YTick',[],'HitTest','off');
ax1e = axes('Position',ax1.Position,'Color','none', 'XTick',[], 'YTick',[],'HitTest','off');
axis(ax1b,'off')
axis(ax1c,'off')
axis(ax1d,'off')
axis(ax1e,'off')


%%
% ---- First legend：for ax1 ----
lgd1 = legend(ax1,[p1r], ...
    {'\textbf{Tendency $\partial_t \langle H \rangle$}'}, ...
    'FontSize',fontsize-1,'Interpreter','latex','Position',[0.1016 0.9175 0.2534 0.0365]);
lgd1.Box = 'off';


%%
% ---- Second legend: ax1b ----
lgd2 = legend(ax1b,[p2r p3r], ...
    {'\textbf{Surface buoyancy flux $\rho_0\langle B_0\rangle/(g\Delta\rho)$}',...
    '\textbf{Ekman buoyancy flux $\rho_0 \langle B_\mathrm{Ek}\rangle/(g\Delta\rho)$}'}, ...
    ...% 'NumColumns', 2,...
    'FontSize',fontsize-1,'Interpreter','latex','Position', [0.4399 0.8839 0.5112 0.075]);
lgd2.Box = 'off';

%%

% ---- Third legend: ax1c ----
lgd3 = legend(ax1c,[p4r], ...
    {'\textbf{Horizontal $\partial_t \langle H \rangle - \rho_0 \langle B_0+B_\mathrm{Ek}\rangle/(g\Delta\rho)$}'}, ...
    'FontSize',fontsize-1,'Interpreter','latex','Position',[0.4249 0.8473 0.5538 0.0365]);
lgd3.Box = 'off';

%%

% ---- Fourth legend: ax1d ----
lgd4 = legend(ax1d,[p7r], ...
    {'\textbf{Diagnosed VBF $\langle-\overline{w^\prime b^\prime}^{z}\rangle$}'}, ...
    'NumColumns', 2,...
    'FontSize',fontsize-1,'Interpreter','latex','Position',   [-0.177 0.5774 0.8545 0.0626]);
lgd4.Box = 'off';


%%
% ---- Fifth legend: ax1e ----
lgd5 = legend(ax1e,[ p5r p6r], ...
    {'\textbf{Parameterized VBF using steric height $\langle- C_e\mu_0g^2 \vert\nabla_h\eta_\mathrm{steric}\vert^2/\vert f\vert\rangle$}',...
    '\textbf{Parameterized VBF using submesoscale SSH $\langle- C_e\mu_0g^2 \vert\nabla_h\eta_\mathrm{submeso}\vert^2/\vert f\vert\rangle$}'}, ...
    'NumColumns', 1,...
    'FontSize',fontsize-1,'Interpreter','latex','Position', [0.0792 0.5236 0.8545 0.0700]);
lgd5.Box = 'off';






%% ============================================================
%%% Panel (b): Reconstructed winter MLD
%%% ============================================================

ax2 = axes('Position',[0.09 0.05+0.06 0.85 0.36-0.06]);
hold(ax2,'on')

tw = datetime(time_rec,'ConvertFrom','datenum');
tw_steric = datetime(time_Hml_recon_steric_2012,'ConvertFrom','datenum');
% tw_wbeddy = datetime(time_Hml_recon_wbeddy_2012,'ConvertFrom','datenum');

h2 = plot(tw_steric, Hml_recon_steric_2012,'-.','Color',brown, 'LineWidth', 2);
h3 = plot(tw_steric, Hml_recon_submeso_2012,'-','Color',olive, 'LineWidth', 2);
% h4 = plot(tw_wbeddy, Hml_recon_wbeddy_2012, 'Color',[0.5 0 0.5], 'LineWidth', 1.8);
h1 = plot(tw, Hml_mean_2012,'-','Color',black, 'LineWidth', 2);

x_event = datetime(2012,1,12);
hvl = plot(ax2, [x_event x_event], [500 700], ':','Color',red, 'LineWidth', 2);
hvl.Color(4) = 0.6; % transparency

grid on; grid minor;box on;
ylabel('\textbf{$\langle H\rangle$  (m)}','Interpreter','latex')

% xlabel('Time')
set(gca,'FontSize', fontsize)
title('Reconstructed mixed layer depth','FontWeight','normal',...
    'FontSize', fontsize+5)

% legend({'Total MLD','Reconstructed steric','Reconstructed SSH submeso','Reconstructed B_{eddy}'}, ...
%        'Location','eastoutside', 'FontSize', 10)

% legend([h1 h2 h3],{'Model-diagnosed MLD \langle\it{H}\rangle','Reconstructed MLD using \eta_{steric}',...
%     'Reconstructed MLD using \eta_{submeso}'}, ...
%         'FontSize', fontsize+2,'Position', [0.2000 0.0838 0.4650 0.1244])

legend([h1 h2 h3],{'\textbf{Model-diagnosed MLD $\langle\it{H}\rangle$}',...
    '\textbf{Reconstructed MLD using $\eta_\mathrm{steric}$}',...
    '\textbf{Reconstructed MLD using $\eta_\mathrm{submeso}$}'}, ...
    'FontSize', fontsize,'Interpreter','latex','Position', [0.1938 0.1263 0.4716 0.1200])
legend('boxoff')

xlim([tw(38) tw(214)])
% xlim([tw(38+24) tw(222)])

% xlim([tw(38+24) tw(214)])
ylim([0 800])



xt = datetime([2012  1 1;
               2012  2 1;
               2012  3 1;
               2012  4 1;
               2012  5 1;
               2012 6 1]);

xticks(xt)
xtickformat('MMM yyyy')



print(gcf, [figdir 'figure3_timeseries'], '-dpng', '-r300')

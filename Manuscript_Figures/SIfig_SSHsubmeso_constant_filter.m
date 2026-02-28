clear;close all;
set(groot, 'DefaultFigureRenderer', 'painters')

load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/timeseries_steric_submesoSSH.mat')

addpath ~/surface_submesoscale/analysis_llc/colormap
load_colors
fontsize = 17;
figdir = '~/surface_submesoscale/Manuscript_Figures/SI_figures/';


time_SSH = datetime(time_SSH, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_SSH_mean = datetime(time_SSH_mean, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_steric = datetime(time_steric, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_submesoSSH = datetime(time_submesoSSH, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_lambda = datetime(time_lambda, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');

t_start = dateshift(min(time_submesoSSH), 'start', 'month');
t_end   = dateshift(max(time_submesoSSH), 'start', 'month') + calmonths(1);
monthly_ticks = t_start : calmonths(1) : t_end;

%%% === Figure 1:  ===
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [52 526 750 900])

for t = monthly_ticks
    xline(t, 'Color', [0.8 0.8 0.8], 'LineStyle', ':', 'LineWidth', 0.5); % light gray lines
end

hold on;


tiledlay = tiledlayout(3,1);
nexttile


% Left y-axis: Gradient magnitude of mixed-layer steric height
yyaxis left
l11 = plot(time_steric, 1e6*eta_steric_grad_mean, 'Color',brown, 'LineWidth', 2);
ylabel('\boldmath$\langle\vert\nabla \eta_\mathrm{steric}\vert\rangle\ (10^{-6}\,\mathrm{m})$','Interpreter','latex');
ax1 = gca;
ax1.YColor =  brown;
set(gca,'FontSize', fontsize);
ylim([0 1.2])

% Right y-axis: Gradient magnitude of submesoscale SSH
yyaxis right
l12 = plot(time_submesoSSH, 1e6*eta_submeso_grad_mean_winter,'Color',olive, 'LineWidth', 2);
ylabel('\boldmath$\langle\vert\nabla\eta_\mathrm{submeso}\vert\rangle\ (10^{-6}\,\mathrm{m})$','Interpreter','latex');
ax1.YColor = olive;  
set(gca,'FontSize', fontsize);
ylim([0 1.2])
grid(ax1, 'on'); grid(ax1, 'minor');
set(gca, 'FontSize', fontsize);
box(ax1, 'on');     

xticks_custom = datetime(2011,11,1) : calmonths(2) : datetime(2012,11,1);
xlim([datetime(2011,11,1), datetime(2012,11,1)]);
xticks(xticks_custom);
xticklabels(datestr(xticks_custom, 'mmm yyyy'));  % or use datestr for full control

title('Submesoscale SSH (constant filter) and steric height','FontWeight','normal',...
    'FontSize', fontsize+4)


nexttile

load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/Hml_tendency_all_plot_data_17km.mat')
tw = datetime(time_rec,'ConvertFrom','datenum');
tw_steric = datetime(time_Hml_recon_steric_2012,'ConvertFrom','datenum');
hold on;
h2 = plot(tw_steric, Hml_recon_steric_2012,'-.','Color',brown, 'LineWidth', 2);
h3 = plot(tw_steric, Hml_recon_submeso_2012,'-','Color',olive, 'LineWidth', 2);
h1 = plot(tw, Hml_mean_2012,'-','Color',black, 'LineWidth', 2);
x_event = datetime(2012,1,12);
hvl = plot([x_event x_event], [500 700], ':','Color',red, 'LineWidth', 2);
hvl.Color(4) = 0.6; % transparency
grid on; grid minor;box on;
ylabel('\textbf{$\langle H\rangle$  (m)}','Interpreter','latex')
set(gca,'FontSize', fontsize)
title('Reconstructed mixed layer depth (constant filter)','FontWeight','normal',...
    'FontSize', fontsize+4)
legend([h1 h2 h3],{'\textbf{Model-diagnosed MLD $\langle\it{H}\rangle$}',...
    '\textbf{Reconstructed MLD using $\eta_\mathrm{steric}$}',...
    '\textbf{Reconstructed MLD using $\eta_\mathrm{submeso}$}'}, ...
    'FontSize', fontsize,'Interpreter','latex','Position', [0.1938 0.1263 0.4716 0.1200])
legend('boxoff')
xlim([tw(38) tw(214)])
ylim([0 900])
xt = datetime([2012  1 1;
               2012  2 1;
               2012  3 1;
               2012  4 1;
               2012  5 1;
               2012 6 1]);
xticks(xt)
xtickformat('MMM yyyy')

legend([h1 h2 h3],{'\textbf{Model-diagnosed MLD $\langle\it{H}\rangle$}',...
    '\textbf{Reconstructed MLD using $\eta_\mathrm{steric}$}',...
    '\textbf{Reconstructed MLD using $\eta_\mathrm{submeso}$ (constant filter)}'}, ...
    'FontSize', fontsize,'Interpreter','latex','Position', [0.165 0.39 0.4716 0.09])
legend('boxoff')




nexttile

fname = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/VHF_timeseries_MapBuoyancyRatio_17km.nc';
VHF_diagnosed = ncread(fname,'Q_eddy_daily');
VHF_steric = ncread(fname,'VHF_steric_fullEOS');
VHF_submeso = ncread(fname,'VHF_submeso_fullEOS');

% alpha_old = 1.6011e-4;
% alpha_new = 2e-4;
% ratio_alpha = alpha_old/alpha_new;
% 
% VHF_steric = ratio_alpha* VHF_steric;
% VHF_submeso = ratio_alpha* VHF_submeso;


time_raw = ncread(fname,'time');
time_units = ncreadatt(fname,'time','units');
ref_date_str = extractAfter(time_units, 'since ');
ref_date = datetime(ref_date_str, 'InputFormat','yyyy-MM-dd');
time = ref_date + days(time_raw);

hold on
l_steric = plot(time, VHF_steric, '-.','Color',brown, 'LineWidth',2);
l_submeso = plot(time, VHF_submeso,'-','Color',olive, 'LineWidth',2);
l_diagnosed = plot(time, VHF_diagnosed,  'k','LineWidth',2);
set(gca,'FontSize', fontsize)
grid on; grid minor; box on;
title('Eddy-induced vertical heat flux (constant filter)','FontWeight','normal',...
    'FontSize', fontsize+5)
ylabel('\boldmath{$\langle\mathrm{VHF}\rangle$ \textbf{(W m$^{-2}$)}}','Interpreter','latex')
lg1= legend([l_diagnosed l_steric l_submeso],'\textbf{Diagnosed VHF $\rho_0 C_p\langle-\overline{w^\prime \theta^\prime}^{z}\rangle$}',...
       '\textbf{Theory using $\eta_\mathrm{steric}$}', ...
       '\textbf{Theory using $\eta_\mathrm{submeso}$ (constant filter)}', ...
       'FontSize', fontsize,'Interpreter','latex','Position',[0.46 0.2 0.3873 0.09]);
lg1.Box = 'off';
xtickformat('yyyy-MM')
xt = datetime([2011 11 1; 
               2012  1 1;
               2012  3 1;
               2012  5 1;
               2012  7 1;
               2012  9 1;
               2012 11 1]);
xticks(xt)
xtickformat('MMM yyyy')
xlim([datetime(2011,11,1) datetime(2012,11,1)])
ylim([-20 430])










annotation('textbox', [0.00 0.95 0.05 0.05], ...
    'String','(a)', 'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.62 0.05 0.05], ...
    'String','(b)',  'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.295 0.05 0.05], ...
    'String','(c)', 'LineStyle','none', ...
    'FontSize',fontsize+4)


tiledlay.Padding = 'compact';



print(gcf, [figdir 'SIfig_SSHsubmeso_constant_filter-matlab'], '-djpeg', '-r300')


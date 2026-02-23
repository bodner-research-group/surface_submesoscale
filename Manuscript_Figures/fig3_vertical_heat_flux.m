%%%% Figure 3 — panel (c)

clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

addpath ~/surface_submesoscale/analysis_llc/colormap
load_colors
fontsize = 19;

% Load data
% fname = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/VHF_timeseries_MeanAlpha.nc';
fname = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/VHF_timeseries_MapBuoyancyRatio.nc';

VHF_diagnosed = ncread(fname,'Q_eddy_daily');
VHF_steric = ncread(fname,'VHF_steric_fullEOS');
VHF_submeso = ncread(fname,'VHF_submeso_fullEOS');


% 
% alpha_old = 1.6011e-4;
% alpha_new = 2e-4;
% ratio_alpha = alpha_old/alpha_new;
% 
% VHF_steric = ratio_alpha* VHF_steric;
% VHF_submeso = ratio_alpha* VHF_submeso;



% Read raw time
time_raw = ncread(fname,'time');
% Read units attribute (e.g. "days since 2011-11-04 00:00:00")
time_units = ncreadatt(fname,'time','units');
% Extract reference date from units string
ref_date_str = extractAfter(time_units, 'since ');
ref_date = datetime(ref_date_str, 'InputFormat','yyyy-MM-dd');
% Convert to actual datetime
time = ref_date + days(time_raw);

fontsize = 19;
figdir = '~/surface_submesoscale/Manuscript_Figures/figure3/';
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [52 526 800 800])
ax2 = axes('Position',[0.09 0.05 0.85 0.3]);
hold on
l_steric = plot(time, VHF_steric, '-.','Color',brown, 'LineWidth',2);
l_submeso = plot(time, VHF_submeso,'-','Color',olive, 'LineWidth',2);
l_diagnosed = plot(time, VHF_diagnosed,  'k','LineWidth',2);

% yline(0,'k--','LineWidth',0.8)

set(gca,'FontSize', fontsize)
grid on; grid minor; box on;


title('Eddy-induced vertical heat flux','FontWeight','normal',...
    'FontSize', fontsize+5)
ylabel('\textbf{VHF (W m$^{-2}$)}','Interpreter','latex')

lg1= legend([l_diagnosed l_steric l_submeso],'\textbf{Diagnosed VHF $\rho_0 C_p\langle-\overline{w^\prime \theta^\prime}^{z}\rangle$}',...
       '\textbf{Theory using $\eta_\mathrm{steric}$}', ...
       '\textbf{Theory using $\eta_\mathrm{submeso}$}', ...
       'FontSize', fontsize,'Interpreter','latex','Position',[0.5127 0.2050 0.3873 0.1200]);
lg1.Box = 'off';


% Match Python-style monthly ticks
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

print(gcf, [figdir 'figure3_VHF'], '-dpng', '-r300')

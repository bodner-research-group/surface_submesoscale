%%%% Figure 3 — panel (c)

clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

addpath ~/surface_submesoscale/analysis_llc/colormap
load_colors
fontsize = 19;

% Load data
fname = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/VHF_timeseries_MapBuoyancyRatio.nc';

VHF_diagnosed = ncread(fname,'Q_eddy_daily');
VHF_steric = ncread(fname,'VHF_steric_fullEOS');
VHF_submeso = ncread(fname,'VHF_submeso_fullEOS');
VHF_steric_linearEOS = ncread(fname,'VHF_steric_linearEOS');
VHF_submeso_linearEOS = ncread(fname,'VHF_submeso_linearEOS');


time_raw = ncread(fname,'time');
time_units = ncreadatt(fname,'time','units');
ref_date_str = extractAfter(time_units, 'since ');
ref_date = datetime(ref_date_str, 'InputFormat','yyyy-MM-dd');
time = ref_date + days(time_raw);

fontsize = 17;
figdir = '~/surface_submesoscale/Manuscript_Figures/SI_figures/';

fg1 = figure('Position',[100 100 800 900],'Color','w'); 
tiledlay = tiledlayout(3,1);



alpha_old = 1.6011e-4;
alpha_new = 2e-04;
ratio_alpha = alpha_old/alpha_new;

VHF_steric_2e4 = ratio_alpha* VHF_steric;
VHF_submeso_2e4 = ratio_alpha* VHF_submeso;
VHF_steric_linearEOS_2e4 = ratio_alpha* VHF_steric_linearEOS;
VHF_submeso_linearEOS_2e4 = ratio_alpha* VHF_submeso_linearEOS;


nexttile

hold on
l_submeso = plot(time, VHF_submeso_2e4,'-','Color',olive, 'LineWidth',2);
l_steric = plot(time, VHF_steric_2e4, '-.','Color',brown, 'LineWidth',2);
l_diagnosed = plot(time, VHF_diagnosed,  'k','LineWidth',2);
l_submeso.Color(4)=0.8;
l_steric.Color(4)=0.8;
l_diagnosed.Color(4)=0.5;
l_steric_linearEOS = plot(time, VHF_steric_linearEOS_2e4, '-.','Color',red, 'LineWidth',2);
l_submeso_linearEOS = plot(time, VHF_submeso_linearEOS_2e4,'-','Color',blue, 'LineWidth',2);

title('Eddy-Induced Vertical Heat Flux','FontWeight','normal',...
    'FontSize', fontsize+6)

set(gca,'FontSize', fontsize)
grid on; grid minor; box on;


% title('Eddy-Induced Vertical Heat Flux','FontWeight','normal',...
%     'FontSize', fontsize+5)
ylabel('\textbf{VHF (W m$^{-2}$)}','Interpreter','latex')




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
ylim([-20 470])


lg1= legend([l_diagnosed l_steric l_submeso l_steric_linearEOS l_submeso_linearEOS],'\textbf{Diagnosed VHF $\rho_0 C_p\langle-\overline{w^\prime \theta^\prime}^{z}\rangle$}',...
       '\textbf{Theory using $\eta_\mathrm{steric}$ (Full EOS)}', ...
       '\textbf{Theory using $\eta_\mathrm{submeso}$ (Full EOS)}', ...
       '\textbf{Theory using $\eta_\mathrm{steric}$ (Linear EOS)}', ...
       '\textbf{Theory using $\eta_\mathrm{submeso}$ (Linear EOS)}', ...
       'FontSize', fontsize,'Interpreter','latex','Position',[0.51 0.78 0.4181 0.16]);
lg1.Box = 'off';



nexttile

hold on
l_submeso = plot(time, VHF_submeso,'-','Color',olive, 'LineWidth',2);
l_steric = plot(time, VHF_steric, '-.','Color',brown, 'LineWidth',2);
l_diagnosed = plot(time, VHF_diagnosed,  'k','LineWidth',2);
l_submeso.Color(4)=0.8;
l_steric.Color(4)=0.8;
l_diagnosed.Color(4)=0.5;
l_steric_linearEOS = plot(time, VHF_steric_linearEOS, '-.','Color',red, 'LineWidth',2);
l_submeso_linearEOS = plot(time, VHF_submeso_linearEOS,'-','Color',blue, 'LineWidth',2);


set(gca,'FontSize', fontsize)
grid on; grid minor; box on;



ylabel('\textbf{VHF (W m$^{-2}$)}','Interpreter','latex')



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
ylim([-20 470])




alpha_old = 1.6011e-4;
alpha_new = 1.41e-04;
ratio_alpha = alpha_old/alpha_new;

VHF_steric = ratio_alpha* VHF_steric;
VHF_submeso = ratio_alpha* VHF_submeso;
VHF_steric_linearEOS = ratio_alpha* VHF_steric_linearEOS;
VHF_submeso_linearEOS = ratio_alpha* VHF_submeso_linearEOS;


nexttile

hold on
l_submeso = plot(time, VHF_submeso,'-','Color',olive, 'LineWidth',2);
l_steric = plot(time, VHF_steric, '-.','Color',brown, 'LineWidth',2);
l_diagnosed = plot(time, VHF_diagnosed,  'k','LineWidth',2);
l_submeso.Color(4)=0.8;
l_steric.Color(4)=0.8;
l_diagnosed.Color(4)=0.5;
l_steric_linearEOS = plot(time, VHF_steric_linearEOS, '-.','Color',red, 'LineWidth',2);
l_submeso_linearEOS = plot(time, VHF_submeso_linearEOS,'-','Color',blue, 'LineWidth',2);


set(gca,'FontSize', fontsize)
grid on; grid minor; box on;


% title('Eddy-Induced Vertical Heat Flux','FontWeight','normal',...
%     'FontSize', fontsize+5)
ylabel('\textbf{VHF (W m$^{-2}$)}','Interpreter','latex')




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
ylim([-20 470])







annotation('textbox', [0.00 0.95-0.014 0.05 0.05], ...
    'String','(a)', 'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.62-0.014 0.05 0.05], ...
    'String','(b)',  'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.295-0.014 0.05 0.05], ...
    'String','(c)', 'LineStyle','none', ...
    'FontSize',fontsize+4)


annotation('textbox', [0.085 0.95-0.04 0.56 0.05],'Color',orange, ...
    'String','\textbf{Typical value} \boldmath{$ \alpha  = 2.0\times 10^{-4}\ ^\circ\mathrm{C^{-1}}$}',  'LineStyle','none', 'Interpreter','latex',...
    'FontSize',fontsize)

annotation('textbox', [0.085 0.62-0.04 0.56 0.05],'Color',orange, ...
    'String','\textbf{Annual mean} \boldmath{$ \alpha  = 1.60\times 10^{-4}\ ^\circ\mathrm{C^{-1}}$}',  'LineStyle','none', 'Interpreter','latex',...
    'FontSize',fontsize)

annotation('textbox', [0.085 0.295-0.045 0.56 0.05],'Color',orange, ...
    'String','\textbf{Jan--Mar mean} \boldmath{$ \alpha  = 1.41\times 10^{-4}\ ^\circ\mathrm{C^{-1}}$}',  'LineStyle','none', 'Interpreter','latex',...
    'FontSize',fontsize)



tiledlay.Padding = 'compact';



print(gcf, [figdir 'SIfig_VHF'], '-djpeg', '-r300')


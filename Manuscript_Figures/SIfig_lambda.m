
clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

fpath = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/';
fname = 'Lambda_MLI_timeseries_daily_surface_reference.nc';
figdir = 'SI_figures/';
fontsize = 17;
load_colors;
% purple = (purple*2+cyan)/3;
purple = blue;

Hml_mean  = ncread([fpath fname],'Hml_mean');
Lambda_MLI_mean  = ncread([fpath fname],'Lambda_MLI_mean');
M2_mean  = ncread([fpath fname],'M2_mean');
M2_mean_new  = ncread([fpath fname],'M2_mean_new');
N2ml_mean  = ncread([fpath fname],'M2_mean_new');

time_raw = double(ncread([fpath fname],'time'));   % days
t0 = datetime(2011,11,1);                  % reference epoch
time = t0 + days(time_raw);

% =====================
% Plot
% =====================
fg1 = figure('Position',[100 100 800 800],'Color','w'); 

tiledlay = tiledlayout(3,1);
nexttile

yyaxis right
plot(time, -Hml_mean, '-.','Color',yellow, 'LineWidth',2)
ylabel('\boldmath$\langle H \rangle\ (\mathrm{m})$','Interpreter','latex')
ax1 = gca;
ax1.YColor = yellow; 

% time_winter = time(62:(62+31+28+13));
% Lambda_MLI_mean_winter = mean(Lambda_MLI_mean(62:(62+31+28+13)),'omitnan')
time_winter = time(22:158);
Lambda_MLI_mean_winter = mean(Lambda_MLI_mean(22:158),'omitnan')


yyaxis left
plot(time, Lambda_MLI_mean/1000, 'Color',blue, 'LineWidth',3)
hold on;
l0 = plot(time_winter, ones(1,length(time_winter))*Lambda_MLI_mean_winter/1000, '-.','Color',purple, 'LineWidth',3);
l0.Color(4) = 0.5;
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\langle \lambda_\mathrm{MLI} \rangle\ (\mathrm{km})$','Interpreter','latex')
title('Most unstable wavelength in the mixed layer','FontSize',fontsize+4,'FontWeight','normal')


annotation('textbox', [0.17 0.76 0.56/2 0.1],'Color',purple, ...
    'String','\textbf{Winter--spring mean:} \ \ \ \ \ \boldmath{$\langle \lambda_\mathrm{MLI} \rangle \approx 17 \,\mathrm{km}$}',  'LineStyle','none', 'Interpreter','latex',...
    'FontSize',fontsize)

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

nexttile
plot(time,M2_mean_new, 'Color',orange4, 'LineWidth',2)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\langle \overline{M^2}^z \rangle\ (\mathrm{s^{-2}})$','Interpreter','latex')
title('Mixed-layer averaged horizontal buoyancy gradient','FontSize',fontsize+4,'FontWeight','normal')
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


nexttile
plot(time,N2ml_mean, 'Color',olive, 'LineWidth',2)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
xlabel('Time','FontSize',fontsize+1)
ylabel('\boldmath$\langle \overline{N^2}^z \rangle\ (\mathrm{s^{-2}})$','Interpreter','latex')
title('Mixed-layer averaged vertical buoyancy gradient','FontSize',fontsize+4,'FontWeight','normal')
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
ylim([0 8]*1e-8)



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


print(gcf, [figdir 'SIfig_lambda'], '-djpeg', '-r300')

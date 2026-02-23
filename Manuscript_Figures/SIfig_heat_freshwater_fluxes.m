
clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

fpath = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/';
fname = 'qnet_fwflx_daily_7day_Bflux.nc';
figdir = 'SI_figures/';
fontsize = 17;
load_colors;

Bflux_7day_smooth  = -ncread([fpath fname],'Bflux_7day_smooth');
Bflux_daily_avg  = -ncread([fpath fname],'Bflux_daily_avg');
fwflx_7day_smooth  = -ncread([fpath fname],'fwflx_7day_smooth');
fwflx_daily_avg  = -ncread([fpath fname],'fwflx_daily_avg');
qnet_7day_smooth  = -ncread([fpath fname],'qnet_7day_smooth');
qnet_daily_avg  = -ncread([fpath fname],'qnet_daily_avg');

time_raw = double(ncread([fpath fname],'time'));   % days
t0 = datetime(2011,11,1);                  % reference epoch
time = t0 + days(time_raw);

% =====================
% Plot
% =====================
fg1 = figure('Position',[100 100 800 800],'Color','w'); 

tiledlay = tiledlayout(3,1);

nexttile
hold on;
plot(time, 0*zeros(1,length(time)), '--','Color','k', 'LineWidth',1)
l1 = plot(time, Bflux_daily_avg, 'Color',red, 'LineWidth',1);l1.Color(4) = 0.5;
l2 = plot(time, Bflux_7day_smooth, 'Color',blue, 'LineWidth',2);
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$-\langle B_0\rangle\ (\mathrm{m^2/s^3})$','Interpreter','latex')
title('Net surface buoyancy input','FontSize',fontsize+4,'FontWeight','normal')
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
legend([l1 l2],'Daily Mean','7-Day Moving Mean','Fontsize',fontsize+1,'Position',  [0.5012 0.7347 0.2812 0.0594])
legend('boxoff')

nexttile
hold on;
plot(time, 0*zeros(1,length(time)), '--','Color','k', 'LineWidth',1)
l1 = plot(time, qnet_daily_avg, 'Color',red, 'LineWidth',1);l1.Color(4) = 0.5;
plot(time, qnet_7day_smooth, 'Color',blue, 'LineWidth',2)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\langle Q_\mathrm{heat} \rangle\ (\mathrm{W/m^2})$','Interpreter','latex')
title('Net surface heat flux','FontSize',fontsize+4,'FontWeight','normal')
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
hold on;
plot(time, 0*zeros(1,length(time)), '--','Color','k', 'LineWidth',1)
l1 = plot(time, fwflx_daily_avg, 'Color',red, 'LineWidth',1);l1.Color(4) = 0.5;
plot(time, fwflx_7day_smooth, 'Color',blue, 'LineWidth',2)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
xlabel('Time','FontSize',fontsize+1)
ylabel('\boldmath$\langle Q_\mathrm{fresh} \rangle\ (\mathrm{kg/m^2/s})$','Interpreter','latex')
title('Net surface freshwater flux','FontSize',fontsize+4,'FontWeight','normal')
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
ylim([-1 3]*1e-4)



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


print(gcf, [figdir 'SIfig_heat_freshwater_fluxes'], '-djpeg', '-r300')

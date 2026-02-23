clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

fpath = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/';
fname = 'Hml_timeseries.nc';
figdir = 'SI_figures/';
fontsize = 17;
load_colors;

dHml_10m_dt  = ncread([fpath fname],'dHml_10m_dt');
dHml_surface_dt  = ncread([fpath fname],'dHml_surface_dt');
Hml_10m_mean  = ncread([fpath fname],'Hml_10m_mean');
Hml_surface_mean  = ncread([fpath fname],'Hml_surface_mean');

time_raw = double(ncread([fpath fname],'time'));   % days
t0 = datetime(2011,11,1);                  % reference epoch
time = t0 + days(time_raw);


% =====================
% Plot
% =====================
fg1 = figure('Position',[100 100 800 600],'Color','w'); 

tiledlay = tiledlayout(2,1);
nexttile

hold on
% yline(zeros(size(time)), '--', 'Color',boxcolor, 'LineWidth',1)
plot(time, Hml_surface_mean, 'Color',blue, 'LineWidth',2)
plot(time, Hml_10m_mean, '-.','Color',orange, 'LineWidth',2)
% ylim([1.2 2.2]*1e-4)
grid on;grid minor;box on;
legend('Surface-referenced MLD','10 m-referenced MLD','Interpreter','latex','FontSize',fontsize+2)
legend('boxoff');
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\langle H \rangle\ (\mathrm{m})$','Interpreter','latex')
title('Domain-averaged mixed layer depth','FontSize',fontsize+4,'FontWeight','normal')
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
hold on
% yline(0, '--', 'Color',boxcolor, 'LineWidth',1)
plot(time,dHml_surface_dt, 'Color',blue, 'LineWidth',2)
plot(time, dHml_10m_dt, '-.','Color',orange, 'LineWidth',2)
% ylim([1.2 2.2]*1e-4)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
xlabel('Time','FontSize',fontsize+1)
ylabel('\boldmath$\partial_t\langle H \rangle\ (\mathrm{m/day})$','Interpreter','latex')
title('Tendency of domain-averaged mixed layer depth','FontSize',fontsize+4,'FontWeight','normal')
legend('Surface-referenced','10 m-referenced','Interpreter','latex','FontSize',fontsize+2)
legend('boxoff');
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


annotation('textbox', [0.00 0.95 0.05 0.05], ...
    'String','(a)', 'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.44 0.05 0.05], ...
    'String','(b)',  'LineStyle','none', ...
    'FontSize',fontsize+4)

tiledlay.Padding = 'compact';


print(gcf, [figdir 'SIfig_Hml_timeseries'], '-djpeg', '-r300')

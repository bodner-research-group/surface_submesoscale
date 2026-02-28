
clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

fpath = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/';
fname = 'KPPhbl_Hml_daily_timeseries.nc';
figdir = 'SI_figures/';
fontsize = 17;
load_colors;
blue = (2*blue + darkblue+gray)/4;
orange = orange3;

Hml  = ncread([fpath fname],'Hml_mean');
KPPhbl  = ncread([fpath fname],'KPPhbl_mean');
time_raw = double(ncread([fpath fname],'time'));   % seconds
t0 = datetime(2011,11,1);                  % reference epoch
time = t0 + days(time_raw);


fname = 'windstress_magnitude_timeseries_daily.nc';
tau = ncread([fpath fname],'tau_mag_mean');
time_raw = double(ncread([fpath fname],'time'));   % seconds
t0 = datetime(2011,9,10);                  % reference epoch
time_wind = t0 + seconds(time_raw);


% =====================
% Plot
% =====================
fg1 = figure('Position',[100 100 700 600],'Color','w'); 

tiledlay = tiledlayout(2,1);
nexttile

hold on
plot(time, KPPhbl, 'Color',orange, 'LineWidth',2)
plot(time, Hml, 'Color',blue, 'LineWidth',2)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$(\mathrm{m})$','Interpreter','latex')
title('KPP boundary depth vs MLD','FontSize',fontsize+4,'FontWeight','normal')
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
% ylim([1.2 2.1]*1e-4)



annotation('textbox', [0.11 0.80 0.2 0.1],'Color',blue, ...
    'String','Mixed layer depth',  'LineStyle','none', ...
    'FontSize',fontsize+3)


annotation('textbox', [0.16 0.576 0.2 0.1],'Color',orange, ...
    'String','KPP boundary layer depth',  'LineStyle','none',...
    'FontSize',fontsize+3)



nexttile
hold on
plot(time_wind, tau,  'Color',orange, 'LineWidth',2)
% ylim([7 8]*1e-4)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
xlabel('Time','FontSize',fontsize+1)
ylabel('\boldmath$ \langle\vert\tau\vert\rangle \ (\mathrm{N/m^2})$','Interpreter','latex')
title('Daily-mean wind stress magnitude','FontSize',fontsize+4,'FontWeight','normal')
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



% tiledlay.TileSpacing = 'compact';
tiledlay.Padding = 'compact';
% AddLetters2Plots(fg1, {'(a)', '(b)'},'FontSize',fontsize+2,...
%     'FontWeight','normal',...
%     'HShift', -0.19, 'VShift', -0.11)


print(gcf, [figdir 'SIfig_KPPhbl_windstress-matlab'], '-djpeg', '-r300')

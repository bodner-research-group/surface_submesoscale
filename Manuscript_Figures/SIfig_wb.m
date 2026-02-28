
clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

fpath = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/';
figdir = 'SI_figures/';
fontsize = 17;
load_colors;

fname = 'hourly_wb_eddy_gaussian_30km_manuscript_timeseries.nc';
wb_eddy_mean  = -ncread([fpath fname],'wb_eddy_mean');

time_raw = double(ncread([fpath fname],'time'));   % hours since reference
t0 = datetime(2011,11,1,0,0,0);                    % reference time
time = t0 + hours(time_raw);                       % convert to datetime

% =====================
% Plot
% =====================
fg1 = figure('Position',[100 100 700 600],'Color','w'); 

tiledlay = tiledlayout(2,1);

nexttile
hold on;
wb_eddy_mean(wb_eddy_mean<-1e-7) = NaN;
% wb_eddy_mean(wb_eddy_mean>2e-8) = NaN;

l3 = plot(time, wb_eddy_mean, 'LineWidth',1);
l3.Color(4)=0.3;

% Convert time to day numbers (ignoring hours)
time_days = dateshift(time, 'start', 'day'); 

% Compute daily mean, ignoring NaNs
[unique_days, ~, idx] = unique(time_days);
daily_mean = accumarray(idx, wb_eddy_mean, [], @mean, NaN);

daily_mean(isnan(daily_mean)) = 0;

% Overlay daily mean
plot(unique_days, daily_mean, 'Color',blue, 'LineWidth', 2)  % red line for daily mean
legend('Hourly', 'Daily mean', 'Location', 'best')


grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\mathrm{VBF}\ (\mathrm{m^2/s^3})$','Interpreter','latex')
title('\boldmath$\mathrm{Eddy\ VBF}\ B_\mathrm{eddy}\! =\! -\overline{(w-\tilde{w})(b-\tilde{b})}^z$','FontSize',fontsize+4,'FontWeight','normal',Interpreter='latex')
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
ylim([-1 0.1]*1e-7)


nexttile
hold on;
l3 = plot(time, wb_eddy_mean,  'LineWidth',2);
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\mathrm{VBF}\ (\mathrm{m^2/s^3})$','Interpreter','latex')
% title('Filter size: 30 km','FontSize',fontsize+4,'FontWeight','normal')

xticks(datetime(2012,1,1):days(10):datetime(2012,3,1))
xtickformat('MMM dd')
ax = gca;
ax.XAxis.SecondaryLabel.Visible = 'off';
xlim([datetime(2012,1,1) datetime(2012,3,1)])
ylim([-11 3]*1e-8)

mean(wb_eddy_mean(61*24:(61+60)*24),'omitnan')


fname = 'hourly_wb_eddy_gaussian_60km_timeseries.nc';
wb_eddy_mean  = -ncread([fpath fname],'wb_eddy_mean');

time_raw = double(ncread([fpath fname],'time'));   % hours since reference
t0 = datetime(2012,1,1,0,0,0);                    % reference time
time = t0 + hours(time_raw);                       % convert to datetime

l3 = plot(time, wb_eddy_mean, 'Color',green, 'LineWidth',2);
% grid on;grid minor;box on;
% set(gca,'FontSize',fontsize)
% ylabel('\boldmath$\mathrm{VBF}\ (\mathrm{m^2/s^3})$','Interpreter','latex')
% title('Filter size: 60 km','FontSize',fontsize+4,'FontWeight','normal')

xticks(datetime(2012,1,1):days(10):datetime(2012,3,1))
xtickformat('MMM dd')
ax = gca;
ax.XAxis.SecondaryLabel.Visible = 'off';
xlim([datetime(2012,1,1) datetime(2012,3,1)])
ylim([-9 0]*1e-8)


mean(wb_eddy_mean)

annotation('textbox', [0.00 0.94 0.05 0.05], ...
    'String','(a)', 'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.44 0.05 0.05], ...
    'String','(b)',  'LineStyle','none', ...
    'FontSize',fontsize+4)



% annotation('textbox', [0.1, 0.55, 0.1, 0.1], ... 
%            'String', '\boldmath$\mathrm{Eddy\ VBF}\ B_\mathrm{eddy}\! =\! -\overline{(w-\tilde{w})(b-\tilde{b})}^z$', ...
%            'Color', blue, ...               
%            'FontSize', fontsize+4,'Interpreter','latex',...
%            'EdgeColor', 'none'); 


annotation('textbox', [0.1, 0.4, 0.1, 0.02], ... 
           'String', '\boldmath$\mathrm{Jan\!\!-\!\!Feb\ Mean\ (30\!\!-\!\!km\ filter):}\ B_\mathrm{eddy}\! =\! -4.24\times10^{-8}\,\mathrm{m^2/s^3}$', ...
           'Color', blue, ...               
           'FontSize', fontsize+3,'Interpreter','latex',...
           'EdgeColor', 'none'); 


annotation('textbox', [0.1, 0.115, 0.1, 0.02], ... 
           'String', '\boldmath$\mathrm{Jan\!\!-\!\!Feb\ Mean\ (60\!\!-\!\!km\ filter):}\ B_\mathrm{eddy}\! =\! -4.85\times10^{-8}\,\mathrm{m^2/s^3}$', ...
           'Color', olive, ...               
           'FontSize', fontsize+3,'Interpreter','latex',...
           'EdgeColor', 'none'); 

tiledlay.Padding = 'compact';


print(gcf, [figdir 'SIfig_wb-matlab_new'], '-djpeg', '-r300')

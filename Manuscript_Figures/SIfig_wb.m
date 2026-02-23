
clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

fpath = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/';
figdir = 'SI_figures/';
fontsize = 17;
load_colors;

fname = 'hourly_wb_eddy_gaussian_30km_manuscript_timeseries.nc';
wb_eddy_mean  = -ncread([fpath fname],'wb_eddy_mean');
wb_mean_mean  = -ncread([fpath fname],'wb_mean_mean');
wb_total_mean  = -ncread([fpath fname],'wb_total_mean');

time_raw = double(ncread([fpath fname],'time'));   % hours since reference
t0 = datetime(2011,11,1,0,0,0);                    % reference time
time = t0 + hours(time_raw);                       % convert to datetime

% =====================
% Plot
% =====================
fg1 = figure('Position',[100 100 700 900],'Color','w'); 

tiledlay = tiledlayout(3,1);

nexttile
hold on;
l1 = plot(time, wb_total_mean, 'Color',blue, 'LineWidth',1);
l2 = plot(time, wb_mean_mean, '-','Color',orange, 'LineWidth',1);
l3 = plot(time, wb_eddy_mean, 'Color',green, 'LineWidth',2);
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\mathrm{VBF}\ (\mathrm{m^2/s^3})$','Interpreter','latex')
title('Filter size: 30 km','FontSize',fontsize+4,'FontWeight','normal')
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
% legend([l1 l2 l3],'Total VBF','Mean VBF', 'Eddy VBF','Fontsize',fontsize+1,'Position',  [0.5012 0.7347 0.2812 0.0594])
% legend('boxoff')
ylim([-2 2]*1e-6)


nexttile
hold on;
l0 = plot(time, 0*wb_total_mean, ':','Color',black, 'LineWidth',1);
l1 = plot(time, wb_total_mean, 'Color',blue, 'LineWidth',1);
l2 = plot(time, wb_mean_mean, 'Color',orange, 'LineWidth',1);
l3 = plot(time, wb_eddy_mean, 'Color',green, 'LineWidth',2);
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\mathrm{VBF}\ (\mathrm{m^2/s^3})$','Interpreter','latex')
title('Filter size: 30 km','FontSize',fontsize+4,'FontWeight','normal')

xticks(datetime(2012,1,1):days(10):datetime(2012,3,1))
xtickformat('MMM dd')
ax = gca;
ax.XAxis.SecondaryLabel.Visible = 'off';
xlim([datetime(2012,1,1) datetime(2012,3,1)])
ylim([-11 3]*1e-8)

mean(wb_eddy_mean(61*24:(61+60)*24),'omitnan')


nexttile
hold on;

fname = 'hourly_wb_eddy_gaussian_60km_timeseries.nc';
wb_eddy_mean  = -ncread([fpath fname],'wb_eddy_mean');
wb_mean_mean  = -ncread([fpath fname],'wb_mean_mean');
wb_total_mean  = -ncread([fpath fname],'wb_total_mean');

time_raw = double(ncread([fpath fname],'time'));   % hours since reference
t0 = datetime(2012,1,1,0,0,0);                    % reference time
time = t0 + hours(time_raw);                       % convert to datetime

l0 = plot(time, 0*wb_total_mean, ':','Color',black, 'LineWidth',1);
l1 = plot(time, wb_total_mean, 'Color',blue, 'LineWidth',1);
l2 = plot(time, wb_mean_mean, 'Color',orange, 'LineWidth',1);
l3 = plot(time, wb_eddy_mean, 'Color',green, 'LineWidth',2);
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$\mathrm{VBF}\ (\mathrm{m^2/s^3})$','Interpreter','latex')
title('Filter size: 60 km','FontSize',fontsize+4,'FontWeight','normal')

xticks(datetime(2012,1,1):days(10):datetime(2012,3,1))
xtickformat('MMM dd')
ax = gca;
ax.XAxis.SecondaryLabel.Visible = 'off';
xlim([datetime(2012,1,1) datetime(2012,3,1)])
ylim([-11 3]*1e-8)


mean(wb_eddy_mean)

annotation('textbox', [0.00 0.95 0.05 0.05], ...
    'String','(a)', 'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.62 0.05 0.05], ...
    'String','(b)',  'LineStyle','none', ...
    'FontSize',fontsize+4)
annotation('textbox', [0.00 0.295 0.05 0.05], ...
    'String','(c)', 'LineStyle','none', ...
    'FontSize',fontsize+4)



annotation('textbox', [0.1, 0.86, 0.1, 0.1], ...   
           'String', '\boldmath$\mathrm{Total\ VBF}\ B_\mathrm{total} = -\overline{\,wb\,}^z$', ...
           'Color', blue, ...              
           'FontSize', fontsize+4,'Interpreter','latex',...
           'EdgeColor', 'none');               
            

annotation('textbox', [0.1, 0.82, 0.1, 0.1], ... 
           'String', '\boldmath$\mathrm{Eddy\ VBF}\ B_\mathrm{eddy}\! =\! -\overline{(w-\tilde{w})(b-\tilde{b})}^z$', ...
           'Color', olive, ...               
           'FontSize', fontsize+4,'Interpreter','latex',...
           'EdgeColor', 'none'); 

annotation('textbox', [0.1, 0.715, 0.1, 0.1], ...  
           'String', '\boldmath$\mathrm{Mean\ VBF}\ B_\mathrm{mean}\! =\! B_\mathrm{total}-B_\mathrm{eddy}$', ...
           'Color', orange, ...               
           'FontSize', fontsize+4,'Interpreter','latex',...
           'EdgeColor', 'none');   


annotation('textbox', [0.3, 0.4, 0.1, 0.02], ... 
           'String', '\boldmath$\mathrm{Jan\!\!-\!\!Feb\ Mean:}\ B_\mathrm{eddy}\! =\! -4.24\times10^{-8}\,\mathrm{m^2/s^3}$', ...
           'Color', olive, ...               
           'FontSize', fontsize+3,'Interpreter','latex',...
           'EdgeColor', 'none'); 


annotation('textbox', [0.3, 0.07, 0.1, 0.02], ... 
           'String', '\boldmath$\mathrm{Jan\!\!-\!\!Feb\ Mean:}\ B_\mathrm{eddy}\! =\! -4.85\times10^{-8}\,\mathrm{m^2/s^3}$', ...
           'Color', olive, ...               
           'FontSize', fontsize+3,'Interpreter','latex',...
           'EdgeColor', 'none'); 

tiledlay.Padding = 'compact';


print(gcf, [figdir 'SIfig_wb-matlab'], '-djpeg', '-r300')

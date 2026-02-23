
clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')

fpath = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/';
fname = 'surfaceT-S-rho-buoy-alpha-beta_gradient_magnitude.nc';
figdir = 'SI_figures/';
fontsize = 17;
load_colors;

alpha  = ncread([fpath fname],'alpha_s_daily');
beta  = ncread([fpath fname],'beta_s_daily');
time_raw = double(ncread([fpath fname],'time'));   % seconds
t0 = datetime(2011,9,10);                  % reference epoch
time = t0 + seconds(time_raw);


% =====================
% Plot
% =====================
fg1 = figure('Position',[100 100 700 600],'Color','w'); 

tiledlay = tiledlayout(2,1);
nexttile

time_winter = time(62:(62+31+28+31));
mean_alpha_winter = mean(alpha(62:(62+31+28+31)),'omitnan');

hold on
plot(time, alpha, 'Color',yellow, 'LineWidth',2)
yline(mean(alpha,'omitnan'), '--', 'Color',olive, 'LineWidth',2.5)
plot(time_winter, ones(1,length(time_winter))*mean_alpha_winter, '--', 'Color',brown, 'LineWidth',2.5)
ylim([1.2 2.2]*1e-4)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
ylabel('\boldmath$ \alpha \ (^\circ\mathrm{C^{-1}})$','Interpreter','latex')
title('Surface thermal expansion coefficient','FontSize',fontsize+4,'FontWeight','normal')
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


annotation('textbox', [0.1+0.032 0.73 0.56 0.05],'Color',olive, ...
    'String','\textbf{Annual mean:} \boldmath{$ \alpha  = 1.60\times 10^{-4}\ ^\circ\mathrm{C^{-1}}$}',  'LineStyle','none', 'Interpreter','latex',...
    'FontSize',fontsize)

annotation('textbox', [0.1+0.01 0.6 0.56 0.05],'Color',brown, ...
    'String','\textbf{Jan--Mar mean:} \boldmath{$ \alpha  = 1.41\times 10^{-4}\ ^\circ\mathrm{C^{-1}}$}',  'LineStyle','none', 'Interpreter','latex',...
    'FontSize',fontsize)

nexttile
hold on
plot(time, beta,  'Color',[0.49 0.18 0.56], 'LineWidth',2)
yline(mean(beta,'omitnan'),  '--', 'Color',[0.49 0.18 0.56], 'LineWidth',1)
ylim([7 8]*1e-4)
grid on;grid minor;box on;
set(gca,'FontSize',fontsize)
xlabel('Time','FontSize',fontsize+1)
ylabel('\boldmath$ \beta \ (\mathrm{kg/g})$','Interpreter','latex')
title('Surface haline contraction coefficient','FontSize',fontsize+4,'FontWeight','normal')
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


print(gcf, [figdir 'SIfig_alpha_beta'], '-djpeg', '-r300')

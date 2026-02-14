clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')
% load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/timeseries_data.mat')
fname = '~/surface_submesoscale/Manuscript_Data/icelandic_basin/TurnerAngle_BuoyancyRatio_daily_Timeseries.nc';
Tu_agreements_new_pct = ncread(fname,'Tu_agreements_new_pct');
Tu_diff_means_new = ncread(fname,'Tu_diff_means_new');

date = ncread(fname,'date');
ref_time = datetime(2011,11,1,0,0,0);   % reference time
time_Tu  = ref_time + days(date);       % convert to datetime

fontsize = 19;

figure(2); clf;
set(gcf, 'Color', 'w', 'Position',  [59 527 1300 280])
set(gca, 'Position', [0.1 0.1100 0.725 0.8150])
hold on;
l21 = plot(time_Tu, Tu_agreements_new_pct,'LineWidth', 2.5);

ylabel('\textbf{\%}','Interpreter','latex');
set(gca,'FontSize', fontsize);
ylim([0 80]); 
grid on;box on;

set(gca, 'FontSize', fontsize);

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

annotation('textbox', [0.4 0.8 0.5 0.1], ...
    'String', 'Percentage of Turner angle differences < 10°', ...
    'FontSize', fontsize+4, ...
    'EdgeColor', 'none');

figdir = '~/surface_submesoscale/Manuscript_Figures/figure1/';

% print(gcf,'-dpng','-r300',[figdir 'fig1_Turner.png']);

clear; close all;
set(groot, 'DefaultFigureRenderer', 'painters')
load('~/surface_submesoscale/Manuscript_Data/icelandic_basin/timeseries_data.mat')

time_Tu = datetime(strtrim(time_Tu), ...
                   'InputFormat','yyyy-MM-dd''T''HH:mm:ss');
fontsize = 19;

% Define start and end of time
t_start = dateshift(min(time_Tu), 'start', 'month');
t_end   = dateshift(max(time_Tu), 'start', 'month') + calmonths(1);

% Generate monthly ticks (every month)
monthly_ticks = t_start : calmonths(1) : t_end;

figure(2); clf;
set(gcf, 'Color', 'w', 'Position',  [59 510 1113 307])
set(gca, 'Position', [0.1 0.1100 0.725 0.8150])

for t = monthly_ticks
    xline(t, 'Color', [0.8 0.8 0.8], 'LineStyle', ':', 'LineWidth', 0.5); % light gray lines
end

hold on;

l21 = plot(time_Tu, Tu_diff_means,'LineWidth', 2.5);
ylabel('\textbf{Mean $\big\vert\mathrm{Tu}_v - \mathrm{Tu}_h\big\vert$}','Interpreter','latex');
set(gca, 'YDir', 'reverse');
set(gca,'FontSize', fontsize);
ylim([0 60]); 

grid on;box on;

% Legends
leg2122 = legend(l21,{'Turner Angle Agreement'},...
    'Location', 'northwest','FontSize',fontsize+1,'Position', [0.1309-0.03 0.7425+0.02 0.3128 0.1632]);
legend('boxoff');

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


figdir = '~/surface_submesoscale/Manuscript_Figures/figure1/';

print(gcf,'-dpng','-r300',[figdir 'fig1_Turner.png']);

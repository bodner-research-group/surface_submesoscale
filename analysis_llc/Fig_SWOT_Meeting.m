clear;close all;

load('~/surface_submesoscale/analysis_llc/figs/icelandic_basin/timeseries_data.mat')

POSITION = [52 526 764 291];
fontsize = 16;

% Convert time strings to datetime
time_qnet = datetime(time_qnet, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
time_Hml = datetime(time_Hml, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');

%% === Figure 1: Air-sea interaction (3 y-axes) ===
figure(1); clf;
set(gcf,'Color','w','Position',POSITION)
yyaxis left
plot(time_qnet, -qnet_7day_smooth, 'b-', 'LineWidth', 2);
ylabel('Q_{net} (W/m^2)');

yyaxis right
plot(time_Hml, squeeze(Hml_mean), 'k-', 'LineWidth', 2);
ylabel('H_{ML} (m)');

% Add third y-axis manually using new axes
ax1 = gca;  % Current axes
set(gca,'FontSize',fontsize);
ax1_pos = ax1.Position;  % Position of first axes
set(gca,'FontSize',fontsize);

% Create new axes for third y-axis (stratification)
ax3 = axes('Position', ax1_pos,...
           'Color','none',...
           'YAxisLocation','right',...
           'XAxisLocation','bottom',...
           'XColor','none', 'YColor','g');
hold(ax3, 'on');
plot(ax3, time_Hml, squeeze(N2ml_mean), 'g-', 'LineWidth', 2);
ylabel(ax3, 'N^2 (s^{-2})');

% Adjust the third y-axis position slightly
ax3.Position(1) = ax3.Position(1) + 0.07;  % move right slightly

% title('Figure 1: Air-Sea Interaction');
% xlabel('Time');
grid on;grid minor;

legend(ax1, {'Q_{net}'}, 'Location', 'northwest');
legend(ax3, {'N^2'}, 'TextColor', 'g', 'Location', 'northeast');
legend(ax1, {'Q_{net}', 'H_{ML}'}, 'Location', 'northwest');

set(gca,'FontSize',fontsize);

%% === Figure 2: Surface diagnostics (2 y-axes) ===
figure(2); clf;
set(gcf,'Color','w','Position',POSITION)

yyaxis left
plot(time_Hml, squeeze(Tu_agreements_pct), 'LineWidth', 2);
ylabel('$\%\ \mathrm{of}\ \vert\mathrm{Tu}_V - \mathrm{Tu}_H\vert \leq 10^\circ$','Interpreter','latex');
ylim([0 100]);

yyaxis right
plot(time_qnet, squeeze(eta_grad_mag_weekly), 'LineWidth', 2);
ylabel('$\vert\nabla\eta\vert$','Interpreter','latex');

grid on;grid minor;
legend({'Turner Angle Agreement', 'SSH gradient magnitude'}, 'Location', 'northwest');

set(gca,'FontSize',fontsize);

%% === Figure 3: Scales (1 y-axis) ===
figure(3); clf;
set(gcf,'Color','w','Position',POSITION)

plot(time_Hml, squeeze(Lambda_MLI_mean), 'r-', 'LineWidth', 2); hold on;
plot(time_Hml, squeeze(Lr_at_max), 'b-', 'LineWidth', 2);
ylabel('Wavelength (km)');
xlabel('Time');
% title('Horizontal Scales');
grid on;grid minor;
legend({'\lambda_{MLI}', 'L_r (wb spectrum)'}, 'Location', 'best');
set(gca,'FontSize',fontsize);

clear all;close all;
addpath colormap/

fname = '/Users/ysi/surface_submesoscale/data_llc/01_gulf/ds_Eta.nc';

ni = ncinfo(fname);
for i=1:length(ni.Variables)
    var_name = ni.Variables(i).Name;
    ds_Eta.(var_name) = ncread(fname, var_name);  % The result is a structure 
end

fields = fieldnames(ds_Eta);

for i = 1:length(fields)
    assignin('base', fields{i}, ds_Eta.(fields{i}));
end

clear ni i var_name ds_Eta fname


fontsize = 17;

for nt = 1:400
figure(1);set(gcf,'Color','w')
pcolor(XC,YC,Eta(:,:,nt));shading interp;colorbar;colormap(WhiteBlueGreenYellowRed(0))
set(gca,'FontSize',fontsize);clim([-0.6 0.6])
end


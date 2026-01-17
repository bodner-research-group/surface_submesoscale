function cmap=comics(ncolors,nstripes)
if nargin==0
  ncolors=128;
end
if nargin<=1
  nstripes=8;
end

x=[0:ncolors-0.2]/ncolors;% to have a colormap from blue to red
cc=zeros(ncolors,3);
cc(:,1)=(1+tanh((x-0.5)*4))/3;
cc(:,2)=1;
cc(:,3)=1-(x-0.5).^2*1.5;

cc(:,2)=cc(:,2).*(1-exp(- ((x'-0.5)/.12).^2 ));
cc(:,3)=cc(:,3).*(1-0.4*sin(x'*pi*nstripes).^40);

cmap=flipud(hsv2rgb(cc));

return

%% to store the palette for ncview
fid=fopen('~/redblue.ncmap','w');
fprintf(fid,'%i %i %i\n',floor(cmap'*255.));
fclose(fid);

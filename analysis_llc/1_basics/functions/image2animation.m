% This program creates a movie/slideshow from a set of images, and save it as an animated GIF file.
% Notice that the quality an image may decrease due to the GIF format.
%
% Written by Moshe Lindner , Bar-Ilan University, Israel.
% September 2010 (C)

clear all
[file_name file_path]=uigetfile({'*.jpeg;*.jpg;*.bmp;*.tif;*.tiff;*.png;*.gif','Image Files (JPEG, BMP, TIFF, PNG and GIF)'},'Select Images','multiselect','on');
% file_name=sort(file_name);

[file_name2 file_path2]=uiputfile('*.gif','Save as animated GIF',file_path);
loops=1;
delay=0.2;
delay1=0.7;
delay2=1;

h = waitbar(0,['0% done'],'name','Progress') ;
for i=1:length(file_name)
    if strcmpi('gif',file_name{i}(end-2:end))
        [M  c_map]=imread([file_path,file_name{i}]);
    else
        a=imread([file_path,file_name{i}]);
        [M  c_map]= rgb2ind(a,256);
    end
    if i==1
        imwrite(M,c_map,[file_path2,file_name2],'gif','LoopCount',loops,'DelayTime',delay1)
    elseif i==length(file_name)
        imwrite(M,c_map,[file_path2,file_name2],'gif','WriteMode','append','DelayTime',delay2)
    else
        imwrite(M,c_map,[file_path2,file_name2],'gif','WriteMode','append','DelayTime',delay)
    end
    waitbar(i/length(file_name),h,[num2str(round(100*i/length(file_name))),'% done']) ;
end
close(h);



addpath ~/surface_submesoscale/Figures_for_Manuscript/colormaps/cmocean/;
addpath ~/surface_submesoscale/Figures_for_Manuscript/colormaps/customcolormap/;

mycolormap = customcolormap(linspace(0,1,11), {'#68011d','#b5172f','#d75f4e','#f7a580','#fedbc9','#f5f9f3','#d5e2f0','#93c5dc','#4295c1','#2265ad','#062e61'});


black = [0 0 0];
verydarkgray = [0.25 0.25 0.25];
darkgray = [0.5 0.5 0.5];
gray = [0.7 0.7 0.7];
boxcolor = [0.85 0.85 0.85];
lightgray = [249 249 249]/255;


red = [0.6350 0.0780 0.1840];
lightred = [249 102 102]/255;
orange = [0.8500 0.3250 0.0980];
coral = [255 127 80]/255;
pink = [255 153 204]/255;

yellow = [0.9290 0.6940 0.1250];
gold = [255 215 0]/255;
brown = [153 102 51]/255;

blue = [0 0.4470 0.7410];
lightblue = [0.3010 0.7450 0.9330];
purple = [0.4940 0.1840 0.5560];

green = [0.4660 0.6740 0.1880];
green2 = [0 153 0]/255;
seagreen = [46 139 87]/255;
olive = [107 142 35]/255;
darkgreen = [21 71 52]/255;

cyan = [0 255 255]/255;

blue1 = '#0047AB';
blue2 = '#97AAFF';
blue3 = '#b8d4ff';
blue4 = '#bbf5ff';
red1 = '#cf513d';
red2 = '#ef7564';
red3 = '#f5d3ce';
bluefill = hex2rgb('#59bfff');
mycolormap_fig2 = [...
hex2rgb(blue1)
hex2rgb(blue2)
hex2rgb(blue3)
hex2rgb(blue4)
black
hex2rgb(red3)
hex2rgb(red2)
hex2rgb(red1)];


BLUE1 = hex2rgb(blue1);
BLUE2 = hex2rgb(blue2);
BLUE3 = hex2rgb(blue3);
BLUE4 = hex2rgb(blue4);
RED1 = hex2rgb(red1);
RED2 = hex2rgb(red2);
RED3 = hex2rgb(red3);

lightpurple = [204 153 255]/255;
purple = [153 51 255]/255;
darkpurple = [102 0 204]/255;
brown1 = [153 76 0]/255;
brown2 = [255 178 103]/255;


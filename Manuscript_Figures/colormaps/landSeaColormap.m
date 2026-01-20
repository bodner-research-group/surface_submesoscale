function h = landSeaColormap(m)
% Set number of steps to user input m. if empty, set it to 200
if nargin < 1
   f = get(groot,'CurrentFigure');
   if isempty(f)
      m = size(get(groot,'DefaultFigureColormap'),1);
   else
      m = 200;
   end
end

seas = fix(11/20*m);
fields = fix(0.5/20*m);
hills = fix(2/20*m);
mountains = fix(6.5/20*m);
r_ = [linspace(0.4,1,mountains)';linspace(1,1,hills)';linspace(1,0,fields)'; linspace(0.53,0,seas)';];
g_ = [linspace(0.15,0.6,mountains)';linspace(0.6,1,hills)';linspace(0.9,0.5,fields)'; linspace(0.8,0,seas)'];
b_ = [linspace(0,0,mountains)';linspace(0,0.8,hills)';linspace(0.8,0,fields)'; linspace(1,0.4,seas)'];
totlength = length(r_);
for i = 1:totlength
    r(i) = r_(totlength-i+1);
    g(i) = g_(totlength-i+1);    
    b(i) = b_(totlength-i+1);
end
h = [r' g' b'];
% Set caxis to caxis([-11000 9000]) to get a good looking map.
caxis([-11000 9000]);

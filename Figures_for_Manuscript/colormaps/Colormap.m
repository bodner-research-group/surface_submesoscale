classdef Colormap
    %COLORMAP perceptually uniform colormaps for oceanography
    %Thyng, K.M., C.A. Greene, R.D. Hetland, H.M. Zimmerle, and S.F. DiMarco. 2016. True colors of oceanography: Guidelines for effective and accurate colormap selection. Oceanography 29(3):9–13. http://dx.doi.org/10.5670/oceanog.2016.66
    %   <a href="https://matplotlib.org/cmocean/">See here for names/images of the colormaps</a>
    
    properties (Constant)
        algae = Colormap.get('algae');
        amp = Colormap.get('amp');
        balance = Colormap.get('balance');
        curl = Colormap.get('curl');
        deep = Colormap.get('deep');
        delta = Colormap.get('delta');
        dense = Colormap.get('dense');
        diff = Colormap.get('diff');
        gray = Colormap.get('gray');
        haline = Colormap.get('haline');
        ice = Colormap.get('ice');
        matter = Colormap.get('matter');
        oxy = Colormap.get('oxy');
        phase = Colormap.get('phase');
        rain = Colormap.get('rain');
        solar = Colormap.get('solar');
        speed = Colormap.get('speed');
        tarn = Colormap.get('tarn');
        tempo = Colormap.get('tempo');
        thermal = Colormap.get('thermal');
        topo = Colormap.get('topo');
        turbid = Colormap.get('turbid');
    end
    
    methods (Static, Access = protected)
        function cmap = get(name)
            folder = fileparts(mfilename('fullpath'));
            C = load(fullfile(folder, 'CMOceanColormaps.mat'), name);
            cmap = C.(name);
        end
    end
    
end


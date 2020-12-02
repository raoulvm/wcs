#from bokeh.plotting import figure

from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from numpy import sin, log, arcsin, exp

#from bokeh.models import DataRange1d, Plot, LinearAxis, Grid, CDSView, GroupFilter, widgets, CustomJS, Button
#from bokeh.models.glyphs import Patches, wedges#, Patch
#from bokeh.io import curdoc, show
#from bokeh.layouts import row, column#, widgetbox
from bokeh import tile_providers
from bokeh.io import output_file, show as bokeh_show
from bokeh.models import HoverTool, LinearColorMapper, LogTicker, ColorBar, SingleIntervalTicker, Text, Legend, Plot,  DataRange1d, LinearAxis, MercatorAxis

import bokeh.models.renderers as rd
from bokeh.models.markers import Circle, Marker
from bokeh.core.enums import LegendLocation
from bokeh.plotting._tools import process_tools_arg
from bokeh.palettes import inferno

from typing import List, Dict

class bokeh_geoplot:
    def __init__(self, tools:str='pan, wheel_zoom, reset', plot_width:int=700, plot_height:int=700, title:str='Geoplot', axes_visible:bool=True, palette:'bokeh.palette.func'=inferno):
        self.tools=tools
        self.plot_width=plot_width
        self.plot_height=plot_height
        self.title=title
        self.axes_visible=axes_visible
        self.fig = None
        self.axes_visible = axes_visible
        self.x_range = [None, None]
        self.y_range = [None, None]
        self.glyphs:Dict[int,Dict[str,object]] = {} # Stores a numbered list (dict) of dict with 'glyph' and 'cds' and 'kwargs' objects
        self.palette = palette
    @staticmethod
    def _lng_to_mercator(lng):
        '''
        lng_to_mercator(lng)
        returns the longitude from polar coordinates in x coordinates of the web mercator projection 
        '''
        return 6378137.0 * lng * 0.017453292519943295
    @staticmethod
    def _lat_to_mercator(lat):
        '''
        lat_to_mercator(lng)
        returns the latitude from polar coordinates in y coordinates of the web mercator projection 
        '''
        north = lat * 0.017453292519943295
        return 3189068.5 * log((1.0 + sin(north)) / (1.0 - sin(north)))
    
    def add_geo_data(self, df:pd.core.frame.DataFrame, lat_field:str, lon_field:str,  legend_label:str='Coordinates',  glyph:Marker=Circle, **glyph_kwargs ):
        # transform mercator, save data to CDS
        cds_dict = dict(y_lat_reserved=self._lat_to_mercator(df[lat_field]),
                        x_lon_reserved=self._lng_to_mercator( df[lon_field]))
        # all other fields
        cds_dict.update({field:df[field] for field in df.columns if field not in [lat_field, lon_field]}) # future use for all fields, can be used in **kwargs for renderer
        cds = ColumnDataSource(cds_dict)
        # change min max of the self.x/z_ranges
        y_min = cds_dict['y_lat_reserved'].min()
        y_max = cds_dict['y_lat_reserved'].max()
        x_min = cds_dict['x_lon_reserved'].min()
        x_max = cds_dict['x_lon_reserved'].max()
        margin = 0.1 # 10% margin
        y_min, y_max = (y_min-(y_max-y_min)*margin,  y_max+(y_max-y_min)*margin ) 
        x_min, x_max = (x_min-(x_max-x_min)*margin,  x_max+(x_max-x_min)*margin ) 

        def safe_min(a,b):
            if a is None:
                return b
            else:
                return min(a,b)
        def safe_max(a,b):
            if a is None:
                return b
            else:
                return max(a,b)

        self.x_range[0] = safe_min(self.x_range[0], x_min)
        self.x_range[1] = safe_max(self.x_range[1], x_max)
        self.y_range[0] = safe_min(self.y_range[0], y_min)
        self.y_range[1] = safe_max(self.y_range[1], y_max)

        # safe the glyph data

        id = len(self.glyphs)
        kwargs = dict(glyph_kwargs)
        if legend_label == 'Coordinates':
            # wasn't changed
            # add numbers if id is >0
            if id>0:
                legend_label = f"{legend_label} ({id+1})"
        kwargs.update({'x':'x_lon_reserved', 'y':'y_lat_reserved'})

        self.glyphs[id] = {'glyph':glyph, 
                           'kwargs':kwargs, 
                           'cds':cds,
                           'legend_label': legend_label}
    
    def show(self):
        # create figure
        self.plot = Plot( 
             x_range=DataRange1d(self.x_range), 
             y_range=DataRange1d(self.y_range), 
             plot_width=self.plot_width, plot_height=self.plot_height, 
              
             )
        self.plot.add_tile(tile_providers.get_provider('CARTODBPOSITRON'))

        #self.plot.add_tools(tools=self.tools)
        tool_objs, tool_map = process_tools_arg(self.plot, self.tools,)
        self.plot.add_tools(*tool_objs)


        if self.axes_visible:
            self.plot.add_layout(MercatorAxis( dimension='lon'), 'below')
            self.plot.add_layout(MercatorAxis( dimension='lat'), 'left')


        #title=self.title,

        palette = self.palette(max(3, len(self.glyphs)))  # get a palette for glyphs with no color spec

        def set_default(d:dict, key, default):
            d.update({key:d.get(key,default)})

        # add data glyphs
        for key,item in self.glyphs.items():
            kwargs = item['kwargs']
            # color, if not provided
            set_default(kwargs, 'fill_color', palette[key])
            set_default(kwargs, 'fill_alpha', 1)
            set_default(kwargs, 'size', (self.x_range[1]-self.x_range[0])/50000)

            # Instanciate the glyph
            glyph = item['glyph'](**kwargs)
            m_kwargs = kwargs.copy()
            set_default(m_kwargs, 'fill_color', palette[key])
            m_kwargs['fill_alpha'] = m_kwargs.get('fill_alpha',1)*0.1 # mute to 10 percent
            set_default(m_kwargs, 'line_alpha', 0)
            muted_glyph = item['glyph'](**m_kwargs)

            item.update(dict(renderer=self.plot.add_glyph(item['cds'], glyph, muted_glyph=muted_glyph)))

        # build the legend
        legend = Legend(
            items=[(item['legend_label'], [item['renderer']]) for key,item in  self.glyphs.items()]
        )
        legend.click_policy="mute"
        legend.title='Click to mute'
        self.plot.add_layout(legend, 'right')

        # io.show()
        output_notebook()
        bokeh_show(self.plot)
                           
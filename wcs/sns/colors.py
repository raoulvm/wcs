import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.cm import register_cmap 

def _rgb_from_decimal(r:int,g:int,b:int)->tuple:
  return (r/255, g/255, b/255)
class Telekom:
  class Colors:
    magenta               = _rgb_from_decimal(226,  0,116)
    class Eye_Catcher:
      eye_catcher_green     = _rgb_from_decimal( 50,185,175)
      eye_catcher_turquise  = _rgb_from_decimal(164,222,238)
      eye_catcher_rose      = _rgb_from_decimal(236,204,191)
      eye_catcher_yellow    = _rgb_from_decimal(240,230,140)
      eye_catcher_blue      = _rgb_from_decimal(  0,168,230)
      eye_catcher_patrol    = _rgb_from_decimal(110,100,140)
    class Strong:
      smaragd   = _rgb_from_decimal(  7,140,130)
      ocean     = _rgb_from_decimal( 90,180,200)
      cappucino = _rgb_from_decimal(189,150,140)
      curry     = _rgb_from_decimal(200,180, 90)
      jeans     = _rgb_from_decimal(  4,120,190)
      aubergine = _rgb_from_decimal( 60, 50, 90)
    class Light:
      mint      = _rgb_from_decimal(134,203,196)
      sky       = _rgb_from_decimal(203,232,244)
      peach     = _rgb_from_decimal(250,226,216)
      vanilla   = _rgb_from_decimal(245,235,175)
      azur      = _rgb_from_decimal( 69,193,241)
      lilac     = _rgb_from_decimal(156,155,186) # "Flieder"     
    class Alarm_Colors:
      # alarm colors
      red       = _rgb_from_decimal(217,  0,  0)
      yellow    = _rgb_from_decimal(254,203,  0)
      green     = _rgb_from_decimal( 70,168,  0)  
    class Greys:
      #greys
      grey38    = _rgb_from_decimal( 38, 38, 38)
      grey75    = _rgb_from_decimal( 75, 75, 75)
      grey115   = _rgb_from_decimal(115,115,115)
      grey178   = _rgb_from_decimal(178,178,178)
      grey220   = _rgb_from_decimal(220,220,220)

  class Cmaps:
    pass

class _cmap_definitions:
    light_green    = (Telekom.Colors.Eye_Catcher.eye_catcher_green,    'white')
    light_turquise = (Telekom.Colors.Eye_Catcher.eye_catcher_turquise, 'white')
    light_yellow   = (Telekom.Colors.Eye_Catcher.eye_catcher_yellow,   'white')
    saturated_green = (_rgb_from_decimal( 38,139,131), _rgb_from_decimal(125,220,214))
    saturated_blue  = (_rgb_from_decimal( 80,191,221), Telekom.Colors.Light.sky)
    saturated_grey  = (_rgb_from_decimal(166,160,186), _rgb_from_decimal( 85, 76,107))
    saturated_yellow= (_rgb_from_decimal(228,212, 57), _rgb_from_decimal(249,245,208))
    two_tones_green_yellow     = (Telekom.Colors.Eye_Catcher.eye_catcher_green,     Telekom.Colors.Eye_Catcher.eye_catcher_yellow)
    two_tones_blue_yellow      = (Telekom.Colors.Eye_Catcher.eye_catcher_blue,      Telekom.Colors.Eye_Catcher.eye_catcher_yellow)
    two_tones_yellow_rose      = (Telekom.Colors.Eye_Catcher.eye_catcher_yellow,    Telekom.Colors.Eye_Catcher.eye_catcher_rose)
    two_tones_turquise_yellow  = (Telekom.Colors.Eye_Catcher.eye_catcher_turquise,  Telekom.Colors.Eye_Catcher.eye_catcher_yellow)    

Telekom.Cmaps._definitions= _cmap_definitions

class _Gradient:
  class Light:
    light_green         = LinearSegmentedColormap.from_list( 'telekom light green',     Telekom.Cmaps._definitions.light_green )
    light_turquise      = LinearSegmentedColormap.from_list( 'telekom light turquise',  Telekom.Cmaps._definitions.light_turquise )
    light_yellow        = LinearSegmentedColormap.from_list( 'telekom light yellow',    Telekom.Cmaps._definitions.light_yellow )
  class Saturated:
    sat_green     = LinearSegmentedColormap.from_list( 'telekom saturated green', Telekom.Cmaps._definitions.saturated_green )
    sat_blue      = LinearSegmentedColormap.from_list( 'telekom saturated blue',  Telekom.Cmaps._definitions.saturated_blue )
    sat_grey      = LinearSegmentedColormap.from_list( 'telekom saturated grey',  Telekom.Cmaps._definitions.saturated_grey )
    sat_yellow    = LinearSegmentedColormap.from_list( 'telekom saturated yellow',Telekom.Cmaps._definitions.saturated_yellow )
  class Two_tones:
    green_yellow    = LinearSegmentedColormap.from_list( 'telekom two tones green yellow',   Telekom.Cmaps._definitions.two_tones_green_yellow )
    blue_yellow     = LinearSegmentedColormap.from_list( 'telekom two tones blue yellow',    Telekom.Cmaps._definitions.two_tones_blue_yellow )
    yellow_rose     = LinearSegmentedColormap.from_list( 'telekom two tones yellow rose',    Telekom.Cmaps._definitions.two_tones_yellow_rose )
    turquise_yellow = LinearSegmentedColormap.from_list( 'telekom two tones turquise yellow',Telekom.Cmaps._definitions.two_tones_turquise_yellow )

Telekom.Colors.Gradient = _Gradient

class _Sequential:
      # build sequential colormaps 
      strong_colors = ListedColormap([Telekom.Colors.Strong.smaragd, 
                                      Telekom.Colors.Strong.ocean,
                                      Telekom.Colors.Strong.cappucino, 
                                      Telekom.Colors.Strong.curry, 
                                      Telekom.Colors.Strong.jeans, 
                                      Telekom.Colors.Strong.aubergine], 
                                    name = 'telekom strong colors')
      light_colors  = ListedColormap([Telekom.Colors.Light.mint, 
                                      Telekom.Colors.Light.sky, 
                                      Telekom.Colors.Light.peach, 
                                      Telekom.Colors.Light.vanilla, 
                                      Telekom.Colors.Light.azur,
                                      Telekom.Colors.Light.lilac], 
                                    name='telekom light colors')
      eye_catchers  = ListedColormap([Telekom.Colors.Eye_Catcher.eye_catcher_blue, 
                                      Telekom.Colors.Eye_Catcher.eye_catcher_green, 
                                      Telekom.Colors.Eye_Catcher.eye_catcher_patrol, 
                                      Telekom.Colors.Eye_Catcher.eye_catcher_rose, 
                                      Telekom.Colors.Eye_Catcher.eye_catcher_turquise, 
                                      Telekom.Colors.Eye_Catcher.eye_catcher_yellow],
                                    name='telekom eye catcher colors')
      greys         = ListedColormap([Telekom.Colors.Greys.grey38, 
                                      Telekom.Colors.Greys.grey75,
                                      Telekom.Colors.Greys.grey115, 
                                      Telekom.Colors.Greys.grey178,
                                      Telekom.Colors.Greys.grey220],
                                      name='telekom greys')

Telekom.Colors.Sequential = _Sequential

def _register():
  '''
  Register Telekom cmaps with matplotlib, so they can be addressed using their string name instead of Telekom.Gradient... or Telekom.Sequential...
  '''
  # register cmaps to be used in matplotlib with their string names
  tcg = Telekom.Colors.Gradient
  tcs = Telekom.Colors.Sequential
  register_cmap(cmap= tcg.Light.light_green)
  register_cmap(cmap= tcg.Light.light_turquise )
  register_cmap(cmap= tcg.Light.light_yellow)
  register_cmap(cmap= tcg.Saturated.sat_green)
  register_cmap(cmap= tcg.Saturated.sat_blue)
  register_cmap(cmap= tcg.Saturated.sat_grey)
  register_cmap(cmap= tcg.Saturated.sat_yellow)
  register_cmap(cmap= tcg.Two_tones.green_yellow)
  register_cmap(cmap= tcg.Two_tones.blue_yellow)
  register_cmap(cmap= tcg.Two_tones.yellow_rose)
  register_cmap(cmap= tcg.Two_tones.turquise_yellow)
  register_cmap(cmap= tcs.strong_colors)
  register_cmap(cmap= tcs.light_colors)
  register_cmap(cmap= tcs.eye_catchers)
  register_cmap(cmap= tcs.greys)

Telekom.register_cmaps = staticmethod(_register)

Telekom.register_cmaps()

def as_palette(cmap, num_colors:int=20)->sns.color_palette:
    """Create a Seaborn Color Palette object (sequential colors) from a MatPlotLib cmap

    Args:
        cmap (cmap): The cmap to use. E.g. Telekom.Colors.Gradient.Two_tones.green_yellow
        num_colors (int, optional): The number of colors the palette object should contain. Defaults to 20.

    Returns:
        sns.color_palette: The color palette created.
    """
    return sns.color_palette(palette=cmap(np.linspace(0,1, num=num_colors)))

def palette_from_list(*colors, num_colors:int=20)->sns.color_palette:
    """Generate a Seaborn color palette from a list of colors (min 2)

    Args:
        num_colors (int, optional): The number of colors the Color Palette should have, will be generated from transitioning one color to the next. Defaults to 20.

    Returns:
        sns.color_palette: The Seaborn Palette generated.
    """
    if len(colors)<2:
        raise ValueError('Need two colors at least')
    cmap =  LinearSegmentedColormap.from_list('', 
        colors = colors
        )(np.linspace(0,1, num=num_colors)) 
    return sns.color_palette(palette=cmap)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import matplotlib.patches as mpatches

def style_document(style='pres'):
    if style=='pres':
        sns.set(style="ticks", context="talk")

        sns.set_style("ticks", {'axes.axisbelow': True,
        'axes.edgecolor': 'white',
        'axes.facecolor': 'black',
        'axes.grid': True,
        'axes.labelcolor': 'white',
        'axes.linewidth': 0,
        'font.family': 'Arial',
        'grid.color': .25,
        'grid.linestyle': '--',
        'image.cmap': 'coolwarm',
        'legend.frameon': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        'lines.solid_capstyle': 'round',
        'figure.facecolor':'black',
        'pdf.fonttype': 42,
        'text.color': 'white',
        'xtick.color': 'white',
        'xtick.direction': 'out',
        'xtick.major.size': 5,
        'xtick.minor.size': 10,
        'ytick.color': 'white',
        'ytick.direction': 'out',
        'ytick.major.size': 5,
        'ytick.minor.size': 10})

       
    elif style=='doc':
        sns.set_style("ticks", {'axes.axisbelow': True,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.labelcolor': 'black',
        'axes.linewidth': 0,
        'font.family': 'Arial',
        'grid.color': 'lightgrey',
        'grid.linestyle': '--',
        'image.cmap': 'coolwarm',
        'legend.frameon': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        'lines.solid_capstyle': 'round',
        'figure.facecolor':'white',
        'text.color': 'black',
        'xtick.color': 'black',
        'xtick.direction': 'out',
        'xtick.major.size': 5,
        'xtick.minor.size': 10,
        'ytick.color': 'black',
        'ytick.direction': 'out',
        'ytick.major.size': 5,
        'ytick.minor.size': 10})
    elif style=='poster':
       sns.set_style("ticks", {'axes.axisbelow': True,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.labelcolor': 'black',
        'axes.linewidth': 0,
        'font.family': 'Arial',
        'grid.color': 'lightgrey',
        'grid.linestyle': '--',
        'image.cmap': 'coolwarm',
        'legend.frameon': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        'lines.solid_capstyle': 'round',
        'xtick.major.width': 2.5,
        'ytick.major.width': 2.5,
        'xtick.minor.width': 2,
        'ytick.minor.width': 2,
        'xtick.major.size': 12,
        'ytick.major.size': 12,
        'xtick.minor.size': 8,
        'ytick.minor.size': 8,
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,})
        

        

    
import matplotlib.colors as mcolor
color_order=["#990000", "#FFCC00", "#7EA172", "#FF7F51",'#995d81','#20BF55','#504136','#247BA0','#DAFF7D','#FFA5AB','#E59F71','#C1DFF0',]
ordinalcmap = mpl.colors.ListedColormap(["#990000", "#FFCC00", "#7EA172", "#FF7F51",'#995d81','#20BF55','#504136','#247BA0','#DAFF7D','#C1DFF0','#FFA5AB','#E59F71'])
norm = mpl.colors.BoundaryNorm(np.arange(-0.5,4), ordinalcmap.N) 

def CustomCmap(from_rgb,to_rgb):
    r1,g1,b1 = from_rgb
    r2,g2,b2 = to_rgb
    r1=r1/255
    r2=r2/255
    g2=g2/255
    g1=g1/255
    b1=b1/255
    b2=b2/255
    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}
    cmap = mcolor.LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

greencmap=CustomCmap((255,255,255),(126, 161, 114))
orangecmap=CustomCmap((255,255,255),(228, 159, 113))
bluecmap=CustomCmap((255,255,255),(36, 123, 160))
redcmap=CustomCmap((255,255,255),(153, 0, 0))
def make_a_clean_num_string(n_tup):
    print(n_tup)
    tups=n_tup[1:-1] .split(',')
    print(tups[0])
    return '('+'{:0.1f}'.format(float(tups[0])).replace('-','')+', '+'{:0.1f}'.format(float(tups[1]))+']'

palette_colors=['#990000','#FFCC00','#7EA172','#247BA0','#995D81','#F49808']
def palettify(columns):
    if type(columns)==list or type(columns)==set or type(columns)==np.array:
        categories=[ordered]
    else:
        categories=columns.unique()
    palette={}
    for x in range(len(categories)):
        print(x)
        palette[categories[x]]=palette_colors[x]
    return palette

from matplotlib.ticker import FixedLocator



style_document('poster')

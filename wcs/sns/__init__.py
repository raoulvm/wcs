from typing import Any, List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def corrheatmap(data:pd.core.frame.DataFrame, 
                vmax:float=1.0,
                diagonal:bool=False,
                decimals:int=2,
                title:str='Correlation Matrix',
                colors:List[str]=['black', 'white', 'black'],
                annot:bool = True,
                as_figure:bool=True,
                figure_params:Dict[Any,Any]={'figsize':(12,6), 'dpi':100},
                ):
    df_corr = data.corr()
    fmt = f'.{decimals}f' if decimals is not None else '.2g'
    k = 1*diagonal
    fig, ax = None, None # satisfy Pylance add in
    if as_figure:
        fig, ax = plt.subplots(**figure_params)
    ax = sns.heatmap(df_corr, 
            cmap = LinearSegmentedColormap.from_list('', colors = colors),
            annot=annot, fmt=fmt, ax=ax, 
            vmin=-vmax, # force min and max for color palette, otherwise 0 might not be in the middle of it!
            vmax=vmax,
            mask = np.triu(np.ones(shape=(len(df_corr),)*2 ) , k=k), # mask the upper half, k=1 leaves the diagonal intact for reference
            )
    if title is not None:
        ax.set_title(title)
    if as_figure:
        return fig
    return ax
    


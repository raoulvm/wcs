
import pandas as pd
import numpy as np
from scipy import stats
#
def df_more_info(df:pd.core.frame.DataFrame): 
    """
    Generate a DataFrame info that is more than pandas DataFrame.info() but less than pandas_profiling.Profile_report(DataFrame)

    Arguments
    =========
    df  DataFrame

    Returns
    -------
    pd.io.formats.style.Styler


    """
    # supposed to be of type pd.io.formats.style.Styler but that has suddenly vanished from colab...

    def highlight_max(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    def highlight_min(s):
        '''
        highlight the minimum in a Series yellow.
        '''
        is_min = s == s.min()
        return ['background-color: lightblue' if v else '' for v in is_min]
    def grey_out_NaN(val, color:str='grey'):
        '''
        text color grey for NaNs in a series
        '''
        #print (val, type(val), val is np.nan)
        return 'color:'+('grey' if val is np.nan else 'black')
    def highlight_object(val, color:str='orange'):
        '''
        '''
        #print (val, type(val), val=='object')
        return 'background-color:'+('orange' if val=='object' else 'white')
    def highlight_nonzero(val, color:str='orange'):
        '''
        '''
        #print (val, type(val), val!=0)
        return 'background-color:'+('orange' if val!=0 else 'white')
    def makestats(col:pd.core.frame.Series)->pd.core.frame.Series:
        def trystat(tryfunc, elsefunc):
            '''
            tryfunc: function (callable)
            elsefunc: function (callable)
            '''
            try:
                val = tryfunc()
            except Exception:
                try:
                    val = elsefunc()
                except Exception:
                    val = np.nan
            return val

        data = {}
        
        data['dtype'] =  col.dtype
        data['nullcount']=      trystat(col.isna().sum,          
                                        lambda: np.nan)
        data['nullcount_frac']= trystat(lambda: col.isna().sum()/len(col), lambda: np.nan)
        data['mean']= col.mean() if isinstance(col.iloc[0], np.number) else np.nan
        data['median']= col.median() if isinstance(col.iloc[0], np.number) else np.nan
        data['std']= col.std(ddof=0) if isinstance(col.iloc[0], np.number) else np.nan
        data['rsd']= abs(col.std(ddof=0)/col.mean()) if isinstance(col.iloc[0], np.number) and col.mean()!=0 else np.nan

        data['mode'] =          trystat(lambda: stats.mode(col).mode[0] if stats.mode(col).count[0]>1 else np.nan, 
                                        lambda: stats.mode(col.astype(str)).mode[0] if stats.mode(col.astype(str)).count[0]>1 else np.nan)
        
        data['mode_count']=     trystat(lambda: stats.mode(col).count[0],
                                        lambda: stats.mode(col.astype(str)).count[0])
        
        data['mode_frac']=      trystat(lambda: stats.mode(col).count[0]/len(col),
                                        lambda: stats.mode(col.astype(str)).count[0]/len(col))
        
        data['min']= col.min() if isinstance(col.iloc[0], np.number) else np.nan
        data['range']= np.ptp(col.dropna(), axis=0) if isinstance(col.iloc[0], np.number) else np.nan
        data['max']= col.max() if isinstance(col.iloc[0], np.number) else np.nan
        data['nUnique']= col.nunique()
        data['nUnique_frac']= col.nunique()/len(col)
        


        return pd.Series(data)

    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('Parameter must be a pandas DataFrame')
    collsall = ['nullcount', 'nullcount_frac', 'dtype' ,'mode','mode_count','mode_frac','mean','std','min','range', 'max', 'nUnique', 'nUnique_frac']

    return df.apply(makestats).T.style\
            .set_properties(**{'text-align':'right', 'background-color':'lightgrey'})\
            .set_precision(2)\
            .applymap(grey_out_NaN, subset=['mode','mode_count','mode_frac','mean','std','min','range', 'max'])\
            .format({pctcol:'{:.1%}' for pctcol in [s for s in collsall if s.endswith('_frac')]})\
            .applymap(lambda x: 'text-align:center; font-style:italic', subset=['mode'])\
            .bar(subset = [s for s in collsall if s.endswith('_frac')], vmin=0, vmax=1)\
            .bar(subset=['rsd'], color='#F0F000')\
            .apply(highlight_max, subset=['min','max','range'])\
            .apply(highlight_min, subset=['min','max','range'])\
            .applymap(highlight_object, subset=['dtype'])\
            .applymap(highlight_nonzero, subset=['nullcount'])
        

# assuming the samples is much bigger than the cities (or clusters or whatever) list.

# two possibilities: cross join and calculate the min distance; iterate and calculate the min for each sample consecutively

# 1 Cross Join

# minimum aggregation, from right to left ==> search the smallest distance for each right_index group

from typing import Tuple, Callable, List, Any
import numpy as np
from pandas.core.frame import DataFrame
from pandas import merge




def fuzzyrightjoin(df_left:DataFrame,
               df_right:DataFrame,
               func: Callable[[Any], float],
               params_left:List[str],
               params_right:List[str],
               func_type:str='Single',
               how:str='smallest',
               threshold:float=None,
               )->DataFrame:
    """performs a RIGHT JOIN between two Pandas DataFrames using a function to be 
    fed with columns from both dataframes, and joining on either:
    * smallest function result
    * largest function result
    * cross join keeping all rows  with a function result above/below a given threshold

    Used for "fuzzy" lookups. Left table is the lookup table.

    Args:
    -----
        df_left (pd.core.frame.DataFrame): left, usually smaller, DataFrame (the lookup table) to join with right dataframe  

        df_right (pd.core.frame.DataFrame): right DataFrame  

        func (Callable[[Any], float]): function to call on columns given in next two parameters  

        params_left (List[str]): column names from left DataFrame to be passed to `func`   

        params_right (List[str]): column names from right DataFrame to be passed to `func`  

        func_type (str, optional): Information on how to feed parameters to `func`,   
            'Single' : call the function for each data record combination  
            'DataFrame': Pass the respective columns as dataframe to the function.    
            Expect about 20 times the run time for 'Single'!  
            Defaults to 'Single'.
            
        how (str, optional): "join condition" referring to the result of `func`   
            Alternatives:  
            'smallest': from all cross join results keep only those with the samllest `func` result per record from the right dataframe. 
            'largest':  from all cross join results keep only those with the largest `func` results per record from the right dataframe. 
            'maximum_threshold':  keep all records with `func` result below or equal to `threshold`  
            'minimum_threshold':  keep all records with `func` result of at least `threshold`
            Defaults to 'smallest'.

        threshold (float, optional): [description]. Defaults to None.

    Raises:
    -------
        NotImplemented: [description]
        ValueError: [description]

    Returns:
    --------
        pd.core.frame.DataFrame: [description]


    Examples
    --------

                Default for searching closest geographical relations between left and right dataframes, 
                to join the "closest" point from left dataframe to each of the records from the right dataframe.
                E.g. df_small is list of cities with coordinates, df_large is customer gps records, and for
                each gps record you want to find the closest city.
                For geo distances you can use wcs.np.geodistances


    """  # DOCSTRING END

    df_cross = merge(left=df_left.assign(dummy_CAFFEEHEX=1).rename_axis('left_index').reset_index(), 
                        right=df_right.assign(dummy_CAFFEEHEX=1).rename_axis('right_index').reset_index(), 
                        how='inner', on='dummy_CAFFEEHEX').drop(columns=['dummy_CAFFEEHEX'])

    params = []
    for p in params_left:
        if p in params_right:
            # duplicate column, will have gotten an a _x attached above
            params += [p+'_x']
        else:
            # unique
            params += [p]

    for p in params_right:
        if p in params_left:
            # duplicate column, will have gotten an a _y attached above
            params += [p+'_y']
        else:
            # unique
            params += [p]

    if func_type.lower() in ['dataframe']:
        df_cross = df_cross.assign(dummy_CAFFEEDISTANCE = func(df_cross[params]))
    elif func_type.lower() in ['single']:
        df_cross = df_cross.assign(dummy_CAFFEEDISTANCE = df_cross[params].apply(lambda rec: func(*rec), axis=1))
    else:
        raise ValueError(f'Parameter error: func_type of "{func_type}"" not in ["DataFrame","Single"]' )

    if how.lower() in ['smallest','nearest','closest','min']:
        return df_cross.groupby('right_index', as_index=False).apply(lambda x: x.iloc[x['dummy_CAFFEEDISTANCE'].argmin()]).drop(columns=['dummy_CAFFEEDISTANCE'])
    elif  how.lower() in ['largest','max']:
        return df_cross.groupby('right_index', as_index=False).apply(lambda x: x.iloc[x['dummy_CAFFEEDISTANCE'].argmax()]).drop(columns=['dummy_CAFFEEDISTANCE'])
    elif  how.lower() in ['minimum_threshold']:
        return df_cross[df_cross['dummy_CAFFEEDISTANCE']>=threshold].drop(columns=['dummy_CAFFEEDISTANCE'])
    elif how.lower() in ['maximum_threshold']:
        return df_cross[df_cross['dummy_CAFFEEDISTANCE']<=threshold].drop(columns=['dummy_CAFFEEDISTANCE'])
    else:
        raise ValueError(f"how={how} not recognized.")

# 2 Iterative (memory optimized)



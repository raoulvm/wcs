import numpy as np
import pandas as pd
# numpy optimized haversine formula for geo distances along great circles
def haversine_df(x:pd.core.frame.DataFrame):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)

    DataFrame must contain for columns of type float with contents: 
    col 0: lon1, 
    col 1: lat1, 
    col 2: lon2, 
    col 3: lat2
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [x.iloc[:,0], x.iloc[:,1], x.iloc[:,2], x.iloc[:,3]])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return (c * r)


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return (c * r)
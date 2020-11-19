import pandas as pd
import numpy as np

from typing import Union, List
def rcat(df:pd.core.frame.DataFrame, 
         numerical_colname: Union[str, List], 
         categorical_colname:Union[str, List]) -> Union[float, pd.core.frame.DataFrame]:
    """
    Calculates a relation ("correlation comparable estimate") between one or
    more categorical and one or more numerical columns in a Pandas Dataframe.

    Args:  
        `df` (pd.core.frame.DataFrame): DataFrame to analyze  
        `numerical_colname` (Union[str, List]): One or more column names, treated as categorial,  
          should not be a continuous (float) column with many unique values.    
        `categorical_colname` (Union[str, List]): One or more column names, treated as categorial,   
          must not be a categorical (object/string) column.    

    Raises:
        TypeError: if df is not a DataFrame, if column name arguments are not str or list.
        KeyError: if pandas cannot find the column names
        ValueError

    Returns:
        Union[float, pd.core.frame.DataFrame]: if column names are both single columns, returns 
          a float (incl. np.inf); if either column name is a list, returns a Pandas Dataframe wit hthe results
    """
    # global type checking
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('df must be a Pandas DataFrame')

    if (isinstance(numerical_colname, list) and  isinstance(categorical_colname, str)) or \
        (isinstance(numerical_colname, str) and isinstance(categorical_colname, list)) or \
        (isinstance(numerical_colname, list) and isinstance(categorical_colname, list)):
        # return a dataframe

        # 
        if not isinstance(numerical_colname, list):
            colnames_num = [numerical_colname]
        else:
            colnames_num = numerical_colname
        if not isinstance(categorical_colname, list):
            colnames_cat = [categorical_colname]
        else:
            colnames_cat = categorical_colname
        
        m_rcat = np.zeros(shape=(len(colnames_cat), len(colnames_num)))
        df_rcat= pd.DataFrame(data = m_rcat, columns=colnames_num, index=colnames_cat)
        for cat in colnames_cat:
            for num in colnames_num:
                df_rcat.loc[cat, num]= rcat(df=df, numerical_colname=num, categorical_colname=cat)
        return df_rcat

    elif (isinstance(numerical_colname, str) and isinstance(categorical_colname, str)):
        # return a single number
        try:
            var_all = df[numerical_colname].var()
            var_i = df[[categorical_colname,numerical_colname]].groupby(categorical_colname)[numerical_colname].var()
            return 1-(var_i.mean()/var_all)
        except ZeroDivisionError as z:
            return np.inf
        except Exception as e:
            raise e
    else:
        raise TypeError('numerical_colname and categorial_colname must be of type "str" or "list of str"')
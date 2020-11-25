import pandas as pd
import numpy as np

from typing import Callable, Union, List, Any, Hashable

def rcat(df:pd.core.frame.DataFrame, 
         numerical_colname: Union[str, List], 
         categorical_colname:Union[str, List],
         groupby_fct:Callable[[Any],Hashable]=None,
         group_by_fct_restrict_numerical:bool=True,
         cardinality_warning:int=100,
         weak_category_warn:int=20) -> Union[float, pd.core.frame.DataFrame]:
    """
    Calculates a relation ("correlation comparable estimate") between one or
    more categorical and one or more numerical columns in a Pandas Dataframe.

    Args:  
        `df` (pd.core.frame.DataFrame): DataFrame to analyze  
        `numerical_colname` (Union[str, List]): One or more column names, treated as categorial,  
          should not be a continuous (float) column with many unique values.    
        `categorical_colname` (Union[str, List]): One or more column names, treated as categorial,   
          must not be a categorical (object/string) column.  
        `groupby_fct` (Callable, Default None): Function to call on the catgorical column, can be used 
          to create bins on continuous data.   
        `group_by_fct_restrict_numerical` (bool, Default True): Only apply above function to numerical 
          columns (i.e. float, int)
        `cardinality_warning` (int, Default 100): Print a warning if more categories in the column.  
        `weak_category_warn` (int, Default 20): Print a warning if a category has less values.  

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
    message_set = set()

    def collmess(new_message):
        nonlocal message_set
        message_set.add(new_message)
    def pukemess():
        if len(message_set)>0:
            for mess in message_set:
                print(mess)

    def __oneResult(numerical_colname: str, categorical_colname:str):
        var_all = df[numerical_colname].var(ddof=0)
        if var_all==0: return 0 # if the variance of the full dataset is 0, no feature can decrease it
        # print(f'{df[categorical_colname].dtype} dtype of col {categorical_colname}')
        if groupby_fct is None or (group_by_fct_restrict_numerical and str(df[categorical_colname].dtype) in ['datetime64[ns]','object','category']): #, 'category']):
            
            if cardinality_warning is not None:
                cardinality = df[categorical_colname].nunique()
                if cardinality>cardinality_warning:
                    collmess(f'Cardinality warning on {categorical_colname}: Column has {cardinality} unique values.')
            if weak_category_warn is not None:
                    lowcat = df[[categorical_colname,numerical_colname]].groupby(categorical_colname)[numerical_colname].apply(len)
                    if (lowcat<weak_category_warn).sum()>0:
                        collmess(f'Column {categorical_colname} contains {(lowcat<weak_category_warn).sum()} categories with less than {weak_category_warn} values.')
            var_i = df[[categorical_colname,numerical_colname]].groupby(categorical_colname)[numerical_colname].var()
        else:
            # there is a function provided to apply to the "catgorical column" - especially if that is not really categorical
            
            if cardinality_warning is not None:
                cardinality = df[[numerical_colname]].assign(**{categorical_colname: df[categorical_colname].apply(groupby_fct)})[categorical_colname].nunique()
                if cardinality>cardinality_warning:
                    collmess(f'Cardinality warning on {categorical_colname}: Column has {cardinality} unique values.')
            if weak_category_warn is not None:
                    lowcat = df[[numerical_colname]].assign(**{categorical_colname: df[categorical_colname].apply(groupby_fct)}).groupby(categorical_colname)[numerical_colname].apply(len)
                    if (lowcat<weak_category_warn).sum()>0:
                        collmess(f'Column {categorical_colname} (transformed) contains {(lowcat<weak_category_warn).sum()} categories with less than {weak_category_warn} values.')

            var_i = df[[numerical_colname]].assign(**{categorical_colname: df[categorical_colname].apply(groupby_fct)}).groupby(categorical_colname)[numerical_colname].var(ddof=0)
        return 1-(var_i.mean()/var_all)


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
                df_rcat.loc[cat, num]= __oneResult( 
                    numerical_colname=num, 
                    categorical_colname=cat)
        pukemess()
        return df_rcat

    elif (isinstance(numerical_colname, str) and isinstance(categorical_colname, str)):
        # return a single number
        
        y = __oneResult(numerical_colname, categorical_colname )
        pukemess()
        return y

    else:
        raise TypeError('numerical_colname and categorial_colname must be of type "str" or "list of str"')







def rwcat(df:pd.core.frame.DataFrame, 
         numerical_colname: Union[str, List], 
         categorical_colname:Union[str, List],
         groupby_fct:Callable[[Any],Hashable]=None,
         group_by_fct_restrict_numerical:bool=True,
         cardinality_warning:int=100,
         weak_category_warn:int=20) -> Union[float, pd.core.frame.DataFrame]:
    """
    Calculates a relation ("correlation comparable estimate") between one or
    more categorical and one or more numerical columns in a Pandas Dataframe.
    The Variances are weighted with the number of samples in each group.

    Args:  
        `df` (pd.core.frame.DataFrame): DataFrame to analyze  
        `numerical_colname` (Union[str, List]): One or more column names, treated as categorial,  
          should not be a continuous (float) column with many unique values.    
        `categorical_colname` (Union[str, List]): One or more column names, treated as categorial,   
          must not be a categorical (object/string) column.  
        `groupby_fct` (Callable, Default None): Function to call on the catgorical column, can be used 
          to create bins on continuous data.   
        `group_by_fct_restrict_numerical` (bool, Default True): Only apply above function to numerical 
          columns (i.e. float, int)
        `cardinality_warning` (int, Default 100): Print a warning if more categories in the column.  
        `weak_category_warn` (int, Default 20): Print a warning if a category has less values.  

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
    message_set = set()

    def collmess(new_message):
        nonlocal message_set
        message_set.add(new_message)
    def pukemess():
        if len(message_set)>0:
            for mess in message_set:
                print(mess)

    def __oneResult(numerical_colname: str, categorical_colname:str):
        count_all = df[numerical_colname].count() # exclude missings
        var_all = df[numerical_colname].var(ddof=0) 
        if var_all==0: return 0 # if the variance of the full dataset is 0, no feature can decrease it
        # print(f'{df[categorical_colname].dtype} dtype of col {categorical_colname}')
        if groupby_fct is None or (group_by_fct_restrict_numerical and str(df[categorical_colname].dtype) in ['datetime64[ns]','object','category']): #, 'category']):
            
            if cardinality_warning is not None:
                cardinality = df[categorical_colname].nunique()
                if cardinality>cardinality_warning:
                    collmess(f'Cardinality warning on {categorical_colname}: Column has {cardinality} unique values.')
            grp = df[[categorical_colname,numerical_colname]].groupby(categorical_colname)[numerical_colname]
            if weak_category_warn is not None:
                    lowcat = grp.apply(len)
                    if (lowcat<weak_category_warn).sum()>0:
                        collmess(f'Column {categorical_colname} contains {(lowcat<weak_category_warn).sum()} categories with less than {weak_category_warn} values.')
            
            var_i = grp.var(ddof=0) * grp.count()
        else:
            # there is a function provided to apply to the "catgorical column" - especially if that is not really categorical
            df_new = df[[numerical_colname]].assign(**{categorical_colname: df[categorical_colname].apply(groupby_fct)})
            if cardinality_warning is not None:
                cardinality = df_new[categorical_colname].nunique()
                if cardinality>cardinality_warning:
                    collmess(f'Cardinality warning on {categorical_colname}: Column has {cardinality} unique values.')
            grp = df_new.groupby(categorical_colname)[numerical_colname]
            if weak_category_warn is not None:
                    lowcat = grp.apply(len)
                    if (lowcat<weak_category_warn).sum()>0:
                        collmess(f'Column {categorical_colname} (transformed) contains {(lowcat<weak_category_warn).sum()} categories with less than {weak_category_warn} values.')

            var_i = grp.var(ddof=0) * grp.count()
        return 1-(sum(var_i)/(var_all*count_all))


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
        #display(df_rcat)
        for cat in colnames_cat:
            for num in colnames_num:
                rwcatval = __oneResult( 
                    numerical_colname=num, 
                    categorical_colname=cat)
                
                df_rcat.loc[cat, num] = rwcatval
        pukemess()
        return df_rcat

    elif (isinstance(numerical_colname, str) and isinstance(categorical_colname, str)):
        # return a single number
        
        y = __oneResult(numerical_colname, categorical_colname )
        pukemess()
        return y

    else:
        raise TypeError('numerical_colname and categorial_colname must be of type "str" or "list of str"')
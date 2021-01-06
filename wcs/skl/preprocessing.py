from sklearn.base import TransformerMixin as _TransformerMixin
from typing import List as _List
import pandas as _pd 
import numpy as _np

class Winsor(_TransformerMixin):
    '''
    Winsorization Transformer
    Extreme values, defined by quantiles, will be set the to quantile value 
    specified.  
    
    Parameters:
    -----------
    q_cut: Quantile (symmetric lower and top) to cut, Values >=0 to <0.5,
            defaults to 2.5%=0.025  

    _Alternative:_   
        q_cut=None         : Set qcut to None, and...
        q_low_cut = float  : Set the low cut margin individually, eg 0.02
                                to cut the lowest 2% of the values,
        q_high_cut = float : Set the high cut margin individually, eg 0.80
                                to cut the top 20% of the values

    '''
    def __init__(self, 
                       q_cut:float=0.025,
                       q_low_cut:float=None,
                       q_high_cut:float=None,
                       name:str="winsor",
                       verbose:bool=False):

        if q_cut is not None and (q_low_cut is not None or q_high_cut is not None):
            raise ValueError('Cannot specify symmetric and asymmetric cut at once.')
        
        self.q_low_cut=q_low_cut if q_low_cut is not None else q_cut
        self.q_high_cut=q_high_cut if q_high_cut is not None else 1-q_cut

        if self.q_low_cut<0 or self.q_high_cut<=self.q_low_cut or self.q_high_cut>1:
            raise ValueError('Quantiles must be float 0.0 ... 1.0 and lower quantile < higher quantile, symmetric quantile spec must be q_cut < 0.5')

        self.name = name
        self.verbose = verbose
        
    def __return_feature_names(self, s:_List=None):
        pname = self.name+'_' if self.name is not None else ''
        if isinstance(s, list) and len(s)>0 and isinstance(s[0], str): 
            return [s[0]+'__'+pname+c for c in self.__features]
        else:
            return [pname+c for c in self.__features]
    def fit(self, X, y=None):
        if self.verbose:
            print(self.name, 'fit() called!')
            print('X is ', type(X))
        if isinstance(X, _pd.core.frame.DataFrame):
            self.__features = X.columns
            self.get_feature_names = self.__return_feature_names
        if len(X.shape)!=2:
            raise ValueError('Data is not 2 dimensional. Either use np.reshape(-1,1) or double square brackets with pd.DataFrames!')

        # fitting: get the quantiles
        self.quantiles_ = _np.quantile(X, q=[self.q_low_cut,self.q_high_cut], axis=0)
        if self.verbose:
            print('Quantiles')
            print(self.quantiles_)

        return self
    def transform(self, X, y=None):
        if self.verbose:
            print(self.name,'transform called!')
            print('X is ', type(X))
        if len(X.shape)!=2:
            raise ValueError('Data is not 2 dimensional. Either use np.reshape(-1,1) or double square brackets with pd.DataFrames!')

        if isinstance(X, _pd.core.frame.DataFrame):
            X2 = X.to_numpy(copy=True)
        else: 
            X2 = X.copy()
        for c in range(X2.shape[1]):
            X2[:,c][X2[:,c]<self.quantiles_[0,c]] = self.quantiles_[0,c]
            X2[:,c][X2[:,c]>self.quantiles_[1,c]] = self.quantiles_[1,c]
        return X2

    def get_params(self, deep:bool=True):
        # required to create unfitted clones
        return {'q_low_cut':self.q_low_cut, 'q_high_cut':self.q_high_cut, 'name':self.name, 'verbose':self.verbose} # return all init-Paramters as dict

    def __repr__(self):
        return f"Winsor(q_low_cut={self.q_low_cut}, q_high_cut={self.q_high_cut}, name={self.name}, verbose={self.verbose}) "

class Winsor_absolute(_TransformerMixin):
    '''
    Winsorization Transformer
    Extreme values, defined by absolutes, will be set the to min max value 
    specified.  
    
    Parameters:
    -----------
    low_cut = float  : Set the low cut margin individually, eg 0.02
                            to cut the lowest 2% of the values,
    high_cut = float : Set the high cut margin individually, eg 0.80
                            to cut the top 20% of the values

    '''
    def __init__(self, 
                       low_cut:float=None,
                       high_cut:float=None,
                       name:str="winsor_absolute",
                       verbose:bool=False):

        self.high_cut = high_cut
        self.low_cut = low_cut
        if self.high_cut is not None and self.low_cut is not None and self.high_cut<=self.low_cut :
            raise ValueError(f'Values must be float and low_cut < high_cut, got {low_cut},{high_cut}')

        self.name = name
        self.verbose = verbose
        
    def __return_feature_names(self, s:_List=None):
        pname = self.name+'_' if self.name is not None else ''
        if isinstance(s, list) and len(s)>0 and isinstance(s[0], str): 
            return [s[0]+'__'+pname+c for c in self.__features]
        else:
            return [pname+c for c in self.__features]
    def fit(self, X, y=None):
        if self.verbose:
            print(self.name, 'fit() called!')
            print('X is ', type(X))
        if isinstance(X, _pd.core.frame.DataFrame):
            self.__features = X.columns
            self.get_feature_names = self.__return_feature_names
        if len(X.shape)!=2:
            raise ValueError('Data is not 2 dimensional. Either use np.reshape(-1,1) or double square brackets with pd.DataFrames!')

        return self
    def transform(self, X, y=None):
        if self.verbose:
            print(self.name,'transform called!')
            print('X is ', type(X))
        if len(X.shape)!=2:
            raise ValueError('Data is not 2 dimensional. Either use np.reshape(-1,1) or double square brackets with pd.DataFrames!')

        if isinstance(X, _pd.core.frame.DataFrame):
            X2 = X.to_numpy(copy=True)
        else: 
            X2 = X.copy()
        for c in range(X2.shape[1]):
            if self.low_cut is not None:
                X2[:,c][X2[:,c]<self.low_cut]  = self.low_cut
            if self.high_cut is not None:
                X2[:,c][X2[:,c]>self.high_cut] = self.high_cut
        return X2

    def get_params(self, deep:bool=True):
        # required to create unfitted clones
        return {'low_cut':self.low_cut, 'high_cut':self.high_cut, 'name':self.name, 'verbose':self.verbose} # return all init-Paramters as dict

    def __repr__(self):
        return f"Winsor_absolute(low_cut={self.low_cut}, high_cut={self.high_cut}, name={self.name}, verbose={self.verbose}) "
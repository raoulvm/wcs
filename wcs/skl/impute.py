from sklearn.base import TransformerMixin
from typing import List
import pandas as pd
import numpy as np
from sklearn import linear_model

class LinRegImputer(TransformerMixin):
    '''
    Linear Regression Imputer
    The linear regression imputer will build a model based on the second column to derive the first column passed. So it needs to get exactly two columns passed!


    '''
    def __init__(self, 
                       low_correlation_threshold:float = 0.6,
                       low_correlation_handling:str='warn', # 'warn', 'raise', 'mute', None
                       name:str="linregimputer",
                       verbose:bool=False):

        if low_correlation_handling is not None and low_correlation_handling not in ['warn', 'raise', 'mute']:
            raise ValueError(f"low_correlation_handling not in ['warn', 'raise', 'mute'] or None. '{low_correlation_handling}' was passed.")

        self.low_correlation_threshold=low_correlation_threshold if low_correlation_threshold is not None else 0
        self.low_correlation_handling=low_correlation_handling if low_correlation_handling is not None else 'mute'

        if self.low_correlation_threshold>1:
            raise ValueError('Pearson correlation can never be >1, can it?')

        self.name = name
        self.verbose = verbose
        
    def __return_feature_names(self, s:List=None):
        pname = self.name+'__' if self.name is not None else ''
        if isinstance(s, list) and len(s)>0 and isinstance(s[0], str): 
            return [s[0]+'__'+pname+self.__features]
        else:
            return [pname+self.__features]
    def fit(self, X, y=None):
        if self.verbose:
            print(self.name, 'fit() called!')
            print('X is ', type(X))
        if isinstance(X, pd.core.frame.DataFrame):
            self.__features = X.columns[0]
            self.get_feature_names = self.__return_feature_names
        if len(X.shape)!=2:
            raise ValueError('Data is not 2 dimensional. Use double square brackets with pd.DataFrames!')
        if X.shape[1]!=2:
            raise ValueError('Data must be exactely 2 columns, one to be imputed and one for the linear regression (independent variable).')
        # fitting: built the lin reg
        if isinstance(X, pd.core.frame.DataFrame):
            Z=X.dropna().astype(float)
        else:
            Z=pd.DataFrame(X).dropna().astype(float)
        pearson = np.corrcoef(Z, rowvar=False)[0,1]
        if pearson<self.low_correlation_threshold:
            if self.low_correlation_handling=='raise':
                raise ValueError(f'Pearson Correlation of {pearson} is between data columns is below {self.low_correlation_threshold} and low_correlation_handling=="{self.low_correlation_handling}"')
            elif self.low_correlation_handling=='warn':
                print((f'Pearson Correlation of {pearson} is between data columns is below {self.low_correlation_threshold} and low_correlation_handling=="{self.low_correlation_handling}"'))
        self.lm = linear_model.LinearRegression().fit(Z.iloc[:,[1]], Z.iloc[:,[0]])

        

        if self.verbose:
            print('Linear Model')
            print(self.lm)

        return self
    def transform(self, X, y=None):
        if self.verbose:
            print(self.name,'transform called!')
            print('X is ', type(X))
        if len(X.shape)!=2:
            raise ValueError('Data is not 2 dimensional. Either use np.reshape(-1,1) or double square brackets with pd.DataFrames!')
        if X.shape[1]!=2:
                raise ValueError('Data must be exactely 2 columns, one to be imputed and one for the linear regression (independent variable).')
        if isinstance(X, pd.core.frame.DataFrame):
            X2 = X.copy()
        else: 
            X2 = pd.DataFrame(X,copy=True)
        if self.verbose:
            print(X2.iloc[:,0].isnull().sum(), "nulls in target column")
            print(X2.iloc[:,1].isnull().sum(), "nulls in source column")
            print(X2.isnull().all(1).sum(), "intersection, not imputable", X2.isnull().all(1).sum()/len(X2))
        
        mask = X2.iloc[:,1].notnull()
        pred = self.lm.predict(X2[mask].iloc[:,[1]])
        ser = pd.DataFrame(pred).iloc[:,0]
        
        X2[X2.columns[0]][mask] = X2[X2.columns[0]][mask].fillna(ser) # boolean filter must be last, otherwise we're working on a copy again!
        return X2.iloc[:,[0]] # return only fixed first column!

    def get_params(self, deep:bool=True):
        # required to create unfitted clones
        return {
            'low_correlation_threshold':self.low_correlation_threshold, 
            'low_correlation_handling' : self.low_correlation_handling, 
            'name':self.name,
            'verbose':self.verbose
         } # return all init-Paramters as dict

    def __repr__(self):
        return f"LinRegImputer(low_correlation_threshold={self.low_correlation_threshold}, low_correlation_handling={self.low_correlation_handling}, name={self.name},verbose={self.verbose}"

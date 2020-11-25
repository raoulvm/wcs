# The idea is to have a class to pass a model to and get a bunch of analyses for free
# 
# 


from typing import Union
import sklearn.pipeline as skpipe
import sklearn.base as skbase
from sklearn.metrics import log_loss as sm_log_loss, accuracy_score as sm_accuracy
import numpy as np
from ..tools import HTMLtable
from .compose import get_feature_names

class Inspector:
    def __init__(self, model:Union[skpipe.Pipeline, skbase.BaseEstimator], X, y_true, X_test=None, y_test=None, caption='Statistics'):
        self.__model = model
        self.__X = X
        self.__y = y_true
        self.__caption = caption
        if X_test is not None and y_test is not None:
            self.__X_test = X_test
            self.__y_test = y_test
        if hasattr(model, 'predict_proba'):
            self.__proba_train = model.predict_proba(self.__X)[:,1]
            if X_test is not None and y_test is not None:
                self.__proba_test = model.predict_proba(self.__X_test)[:,1]
        if hasattr(model, 'predict'):
            self.__pred_y = model.predict(self.__X)
            if X_test is not None and y_test is not None:
                self.__pred_test = model.predict(self.__X_test)
        
    @property
    def log_loss(self):
        if hasattr(self, '__logloss'):
            # cached values
            return self.__logloss

        if hasattr(self, '__proba_train'):
            train =  sm_log_loss(self.__y, self.__proba_train)
            test = np.nan
            if hasattr(self, '__proba_test'):
                test = sm_log_loss(self.__y_test, self.__proba_test)
            self.__logloss = (train, test)
            return (train, test)
        else:
            self.__logloss = (np.nan, np.nan)
            return (np.nan, np.nan)

    @property 
    def accuracy(self):
        if hasattr(self, '__accuracy'):
            return self.__accuracy

        if hasattr(self, '__pred_y'):
            train = sm_accuracy(self.__y, self.__pred_y)
            test = np.nan
            if hasattr(self, '__pred_test'):
                test = sm_accuracy(self.__y_test, self.__pred_test)
            self.__accuracy = (train, test)
            return self.__accuracy
        else:
            return (np.nan, np.nan)

    def _repr_html_(self):
        tab = HTMLtable(rows = 3, cols=3, caption=self.__caption)
        tab[0,2] = 'Train Set'
        tab[0,3] = 'Test Set'
        tab[1,0] = 'Accuracy'
        tab[1,1] = f'{self.accuracy[0]:.2%}'
        tab[1,2] = f'{self.accuracy[1]:.2%}' 
        tab[2,0] = 'Log Loss'
        tab[2,1] = f'{self.log_loss[0]:.4f}'
        tab[2,2] = f'{self.log_loss[1]:.4f}'
        return tab._repr_html_()

    def __repr__(self) -> str:
        return f'Class wcs.skl.model_analyzer.Inspector(...)'
    
    @property
    def columns(self):
        if hasattr(self, '__columns'):
            #cached value
            return self.__columns
        try:
            cols = get_feature_names(self.__model, list(self.__X.columns))
            self.__columns = cols
            return self.columns
        except Exception:
            raise TypeError('Cannot acquire column names from model and X')



    
    
    
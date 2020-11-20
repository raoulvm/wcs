from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from typing import List

def _inspect_pipeline(p:Pipeline, input_colnames:List, tname:str='')->List:
    # iterate through the pipeline steps to check if any steps change the column list
    # print('Begin Pipeline ', tname)
    cols = input_colnames.copy()
    for st in p.steps:
        #print(st[0], type(st[1]))
        if isinstance(st[1], Pipeline):
            cols = _inspect_pipeline(st[1], cols)
        elif isinstance(st[1], ColumnTransformer):
            # a column transformer probably resets the column list completely (to be confirmed!) 
            cols = _inspect_col_transformer(st[1])
        else:
            cols = _inspect_encoder(st[1], cols)
    # print(f'End Pipeline {tname}- ', cols)
    return cols

def _inspect_encoder(enc, column_list:List, tname:str='')->List:
    # print(f'Begin Encoder {tname}')
    if hasattr(enc, 'get_feature_names'): # can return the feature names
        #if isinstance(enc, (OneHotEncoder, PolynomialFeatures, )): # creates multiple columns
        # print('One Hot Columns:') # just for debugging
        # print(enc.get_feature_names(column_list)) # debug
        # print(f'End Encoder {tname} - ', enc.get_feature_names(column_list))
        if column_list is None or len(column_list)==0:
            return enc.get_feature_names([tname])
        return enc.get_feature_names(column_list) # get input columns X features from train data
    else:
        # print(f'End Encoder {tname} - ', column_list)
        # print(tname, enc.get_params())
        return [tname+'__'+c for c in column_list] # just use the input column(s)
        #print('Columns', trs[2]) # debug

def _inspect_featureunion(fu:FeatureUnion, column_list:List, tname:str=''):
    collist = []
    for trs in fu.transformer_list:
        collist.extend(_inspect_transformer(trs[1],column_list, trs[0]))
    return collist



def _inspect_transformer(transformer, columns_passed, tr_name:str,ct_remainder:str='drop'):

    # meta transformer
    if isinstance(transformer, Pipeline):
        return _inspect_pipeline(transformer, columns_passed, tr_name)
    elif isinstance(transformer, FeatureUnion):
        return _inspect_featureunion(transformer, columns_passed, tr_name)
    elif isinstance(transformer, ColumnTransformer):
        # a column transformer probably resets the column list completely (to be confirmed!) if not part of a pipeline
        return _inspect_col_transformer(transformer)
   
    # single transformer
    else:
        return _inspect_encoder(transformer,columns_passed, tr_name)

def _inspect_col_transformer(fitted_col_transform:ColumnTransformer, tname:str='')->List:
    # print('Begin ColTransformer ', tname)
    collist = [] # my object to collect the column names
    for trs in fitted_col_transform.transformers_:
        #print('Transformer:', trs) # just for debugging!

        # deal with remainder
        if fitted_col_transform.remainder=='drop' and trs[3]=='remainder':
            # do not add anything, because the the remaining columns get dropped
            pass
        elif fitted_col_transform.remainder=='passthrough' and trs[3]=='remainder':
            #
            print(trs)
            raise NotImplementedError(f'Determination of remainder="passthrough" has not been implemented yet. Error in {tname}')
            #

        # deal with column passthrough/drop notation for specific columns
        elif isinstance(trs[1], str)  and trs[1]=='passthrough':
            collist.extend(trs[2]) # just keep the column(s)
        elif isinstance(trs[1], str)  and trs[1]=='drop':
            collist.extend(trs[2]) # just keep the column(s)

        # everything else
        else:
            collist.extend(_inspect_transformer(trs[1],trs[2], trs[0]), fitted_col_transform.remainder)
    #print(f'End ColTransformer - {tname} ',collist)
    return collist

def get_feature_names(sklearn_object, input_features:List=[], )->List[str]:
    """returns the feature names created by `ColumnTransformer`s, `Pipeline`s and `FeatureUnion`s, and standard transformers, also encapsulated in each others.

    Args:
        sklearn_object ([type]): The Transformer to analze
        input_features (List, optional): For Transformers that do not have original feature names, pass the names you want them to have, e.g. if zou are using numpy arrays as input for a pipeline. Defaults to [].

    Returns:
        List[str]: Feature names of the Transfromation.
    """
    if isinstance(sklearn_object, Pipeline):
        return _inspect_pipeline(sklearn_object, input_colnames=input_features )
    elif isinstance(sklearn_object, FeatureUnion):
        return _inspect_featureunion(sklearn_object, column_list=input_features)
    elif isinstance(sklearn_object, ColumnTransformer):
        return _inspect_col_transformer(sklearn_object)
    else:
        # pass on
        return sklearn_object.get_feature_names(input_features)
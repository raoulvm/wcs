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
    if not hasattr(fitted_col_transform, 'transformers_'):
        raise ValueError('ColumnTranformer not fitted yet!')
    for trs in fitted_col_transform.transformers_:
        #print('Transformer:', trs) # just for debugging!

        # deal with remainder
        if fitted_col_transform.remainder=='drop' and trs[0]=='remainder':
            # do not add anything, because the the remaining columns get dropped
            pass
        elif fitted_col_transform.remainder=='passthrough' and trs[0]=='remainder':
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
            collist.extend(_inspect_transformer(trs[1],trs[2], trs[0]))
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


#######################################

# need to get the transforms in order using pipelines




def repipe_transformer_tuples(column_rules:list, withnames:bool=True)->list:
    '''
    Parses a transformer-tuple list object and inserts pipelines if multiple operations are found for one column
    [('transformation_name1', func1, ['col_A', 'col_B']),  
     ('transformation_name2', func2, ['col_A', 'col_C'])]  
     
    will be returned as  
    [('col_A', Pipeline([('transformation_name1', func1,),('transformation_name2', func2,)]), 'col_A'),
     ('col_B#transformation_name1', func1, ['col_B']),  
     ('col_C#transformation_name2', func2, ['col_C'])]  

    if `withnames` is False, the argument list must not contain the name part in the tuples (only func and cols), 
    and the function will also not return names in the tuples:

    [(func1, ['col_A', 'col_B']),  
     (func2, ['col_A', 'col_C'])]  
     
    will be returned as  
    [(Pipeline([('tn1', func1,),('tn2', func2,)]), 'col_A'),
     (func1, ['col_B']),  
     ( func2, ['col_C'])]  

    '''
    if withnames:
        CT_TUPLE_NAME   = 0
        CT_TUPLE_FUNC   = 1
        CT_TUPLE_FIELDS = 2
    else:
        CT_TUPLE_NAME = None
        CT_TUPLE_FUNC   = 0
        CT_TUPLE_FIELDS = 1       
    PI_TUPLE_KEY   = 0
    PI_TUPLE_VALUE   = 1
    pipe_dict_rules = {}
    for rule in column_rules:
        #print(rule)
        # separate the fields
        for field in rule[CT_TUPLE_FIELDS]:
            if withnames:
                pipe_dict_rules[field] = pipe_dict_rules.get(field,[]) + [(rule[CT_TUPLE_NAME], rule[CT_TUPLE_FUNC],)]
            else:
                pipe_dict_rules[field] = pipe_dict_rules.get(field,[]) + [(rule[CT_TUPLE_FUNC], )]

    # create PipeLines
    transformer_list = []
    for pipe in pipe_dict_rules.items():
        if len(pipe[PI_TUPLE_VALUE])==1:
            # only one rule to apply
            if withnames:
                transformer_list.append((
                    pipe[PI_TUPLE_KEY] + '#' + pipe[PI_TUPLE_VALUE][0][CT_TUPLE_NAME],
                    pipe[PI_TUPLE_VALUE][0][CT_TUPLE_FUNC],
                    [pipe[PI_TUPLE_KEY]], # name of pipeline is the field name
                ))
            else:
                transformer_list.append((
                    pipe[PI_TUPLE_VALUE][0][CT_TUPLE_FUNC],
                    [pipe[PI_TUPLE_KEY]], # name of pipeline is the field name
                ))                
        else:
            # multiple rules to put in pipeline object
            if withnames:
                transformer_list.append((
                    pipe[PI_TUPLE_KEY],
                    Pipeline([(name, trnsf) for (name, trnsf) in pipe[PI_TUPLE_VALUE]
                            ]),
                    [pipe[PI_TUPLE_KEY]], # name of pipeline is the field name        
                ))
            else:
                transformer_list.append((
                    Pipeline([('tn1'+str(num), trnsf[0]) for (num, trnsf) in enumerate(pipe[PI_TUPLE_VALUE])
                            ]),
                    [pipe[PI_TUPLE_KEY]], # name of pipeline is the field name        
                ))                
    return transformer_list
# IMPROVEMENT IDEA: COLLATE IDENTICAL TRANSORMATIONS TO LIST OF COLUMNS AGAIN



######################################

def make_transformer_list(tlist:list, withnames:bool=True)->list:
    """
    Generate (raw) list of Transformers for sklearn.compose.ColumnTransformer.  
    The transformers in the list will get instatiated before returning, so they are all brand new!  
    Do not nest in Pipelines, that is incompatible here.  
    You can pass the result to `wcs.skl.compose.repipe_transformer_list()` to get the transformers for one field encapsulated in a pipeline, each.

    arguments
    ---------
    takes a list of transformers in the form [( transformer-class, {parameters for instantiation}, [columns] )]

    * `tranformer-class` must not be initialized/instantiated, so OrdinalEncoder 
    is fine OrdinalEncoder() is not
    * `{parameters for instantiation}` can be None or {} if no arguments are required
    * `[columns]` can be string or list of strings, will be passed on as list

    if `withnames` is True (default) a unique name (col+Number) will be created 
    for ColumnTransformer, False for make_column_transformer() which creates names itself.

    """
    result = []
    i = 0
    for t in tlist:
        cls = t[0]
        params = {} if t[1] is None else t[1]
        cols = [t[2]] if isinstance(t[2], str) else t[2] # just pass on if it is a list or an object other than str
        if isinstance(cls, str):
            pass # can be 'passthrough' or 'drop'
        else:
            cls = cls(**params)
        if withnames:
            result.append(( str(cols).replace(' ','').replace('[','').replace(']','').replace("'",'')+str(i),
                           cls,
                           cols))
        else:
            result.append((cls,
                           cols))            
        i += 1
    return result
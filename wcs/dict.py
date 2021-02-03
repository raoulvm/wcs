def reverse_dict(mapping:dict, always_list:bool=True)->dict:
    """Switches keys and values of a dictionary.
    As values can be duplicates, keys can and up in lists. If always_list==True,
    former keys are always encapsulated in lists even if the values are unique

    Args:
        mapping (dict): The dict to reverse
        always_list (bool, optional): If True the dict contains all values as lists. Defaults to True.

    Returns:
        dict

    Example:

    reverse_dict({'A':1, 'B': 1, 'C':2}, always_list=False)
    >>> {1:['A','B'], 2:'C'}
    
    reverse_dict({'A':1, 'B': 1, 'C':2}, always_list=True)
    >>> {1:['A','B'], 2:['C']}

    """
    rev_mapping = {}
    list_elements = []
    for (k,v) in mapping.items():
        #print(k,v)
        if v not in rev_mapping.keys():
            if always_list:
                rev_mapping.update({v:[k]})
            else:
                rev_mapping.update({v:k})
        else:
            if v in list_elements:
                rev_mapping[v].append(k)
            else:
                rev_mapping[v] = [rev_mapping[v],k]
                list_elements += [v]
    return rev_mapping


def append_dict(dic_a:dict, dic_b:dict):
    """On the fly add/change a key of a dict and return a new dict, without changing the original dict
    Requires Python 3.5 or higher.
    In Python 3.9 that can be substituted by 
    
    dic_a | dic_b

    Args:
        dic_a (dict): [description]
        dic_b (dict): [description]

    Returns:
        [type]: [description]
    """
    return {**dic_a, **dic_b} # Python 3.5+


def walkthrough(dict_of_lists:dict)->list:
    """List generator for GridSearch-like parameter searches


    >>>walkthrough({'a':[1],'b':5, 'c':[3,6],'d':[1,2,3,4]})
    [1] Iterating 8 item combinations
        [{'a': 1, 'b': 5, 'c': 3, 'd': 1},
        {'a': 1, 'b': 5, 'c': 3, 'd': 2},
        {'a': 1, 'b': 5, 'c': 3, 'd': 3},
        {'a': 1, 'b': 5, 'c': 3, 'd': 4},
        {'a': 1, 'b': 5, 'c': 6, 'd': 1},
        {'a': 1, 'b': 5, 'c': 6, 'd': 2},
        {'a': 1, 'b': 5, 'c': 6, 'd': 3},
        {'a': 1, 'b': 5, 'c': 6, 'd': 4}]

    Args:
        dict_of_lists (dict): The dictionary with lists as values. (Scalars will be converted)

    Returns:
        list: A list of dictionaries. For each dict another combination from the lists of the original dict is used.
    """
    def crossmul(dic_of_lists:dict)->int:
        res = 1
        for tup in dic_of_lists.items():
            if isinstance(tup[1], list):
                res *= len(tup[1])
        return res
    def __iterate(the_list:list, __cur=[]):
        __debug=False
        def _print(*arg, **kwargs):
            if __debug:
                print(*args, **kwargs)
        par = the_list[0][0]
        _print(__cur) #, end=',')
        pvm = []
        if not isinstance(the_list[0][1], list): # avoid single items breaking th e tool
            the_list[0][1] = [the_list[0][1]]

        if len(the_list)>1: # there are more parameters to come
            for pv in the_list[0][1]:
                pvm.extend( __iterate(the_list[1:], __cur + [(par, pv)]) )# go through the others
        else: 
            # lowest level
            for pv in the_list[0][1]:
                _print('cur= ', __cur)
                _print('appending', __cur + [(par, pv)])
                pvm.append(__cur + [(par, pv)])
        _print('returning', pvm)
        return pvm
    print('Iterating', crossmul(dict_of_lists), 'item combinations')
    tups = [[k,v] for (k,v) in dict_of_lists.items()]
    return [dict(a) for a in __iterate(tups)]
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

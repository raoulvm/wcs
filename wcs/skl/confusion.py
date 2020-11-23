import numpy as np
from ..tools import HTMLtable
from IPython.display import display
from .metrics import pretty_confusionmatrix as new_location

def pretty_confusionmatrix(*args, **kwargs)->None:
    """Deprecation warning: wcs.skl.confusion.pretty_confusionmatrix() has been moved to wcs.skl.metrics.pretty_confusionmatrix() \n Just to confuse you even more.

    """
    print('Deprecation warning: wcs.skl.confusion.pretty_confusionmatrix() has been moved to wcs.skl.metrics.pretty_confusionmatrix() \n Just to confuse you even more.')
    return new_location(*args, **kwargs)
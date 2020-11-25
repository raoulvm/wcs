# wcs
wcs training tools

for Google Colaboratory

# install and import:
```python
!pip install "git+http://github.com/snowdd1/wcs.git" --upgrade >/dev/null
# e.g.: import google drive share helper
from wcs.google import google_drive_share
```


# contains
## Google Tools
### wcs.google.google_drive_share
*Helps with loading files from google drive links. See docString*

## SciKit Learn Tools
### wcs.skl.metrics.pretty_confusionmatrix()
*can print nicer explainable confusion matrices. pass it a confusion matrix and enjoy.* **Work In Progress Warning**  
**new location since v. 0.0.17**
```python
pretty_confusionmatrix(confusionmatrix: np.ndarray, textlabels:List[str]=['Positive','Negative'], title:str='Confusion Matrix', texthint:str='', metrics:bool=True)->Union[object, dict]:
    """Create a more readable HTML based confusion matrix, based on sklearn 

    Args:
        confusionmatrix (np.ndarray): a sklearn.metrics.confusionmatrix  
        textlabels (List[str], optional): The class labels as list of strings. 
            Defaults to ['Positive','Negative'].  
        title (str, optional): The confusion matrix' title. Defaults to 'Confusion Matrix'.
        texthint (str, optional): Text to print in the top left corner. Defaults to ''. 
            If an empty string (default) is passed, print the population number.  
        metrics (bool, optional): Print the confusion matrix immediately, and return a 
            dict with the metrices. Defaults to True. If set to False, the function 
                returns the confusion metrix as HTMLTable object.

    Returns:
        Union[HTMLTable, dict]: The matrix as HTMLTable if `metrics` is set to False, a dict with the metrics otherwise (Default)
    """
```
### wcs.skl.compose.get_feature_names()
*returns the output columns of e.g. a column transformer with nested pipelines*

### wcs.skl.compose.repipe_transformer_tuples()
*collates transformations for the same columns into Pipelines. See DocString*
Caveat: Do **not** use if you have transformations that require multiple columns to be passed at once! The "re-piper" will 
break them into multiple calls, for each column one call.

### wcs.skl.compose.make_transformer_list(tlist:list, withnames:bool=True)->list:
*instantiates transformers for multiple use of the transformation list without the need of resetting them again*

### wcs.skl.model_selection.train_test_split()
*wraps `sklearn.model_selection.train_test_split()` so the indices get all reset before returning the data*

### wcs.skl.rcat()
#### Mass fuer die "Korrelation" zwischen einer numerischen und einer kategoriellen Variable



Voraussetzungen
* Varianz der numerischen Variable $var_{num} \neq 0$ und $var_{num} \neq \infty$
* Varianz $ var_{i} $ von $n$ Teilmengen berechenbar und niemals $\infty$

$$ R_{cat} = 1-\frac{\sum_{i=0}^n{var_{i}}}{n \cdot var_{num}}  $$

Die Aufteilung in $n$ Teilmengen erfolgt anhand einer zweiten, kategoriellen Variable.

Aussage: Um wieviel nimmt die Varianz ab, wenn ich die kontinuierliche Variable anhand der kategoriellen in einzelne Gruppen zerlege?

## NumPy tools
### wcs.np.print_matrix
*nicer printout of 1 and 2-dimensional matrices in colab, can also print some matrix properties. See DocString*

## Seaborn and MatplotLib tools
### wcs.sns.corrheatmap
Print a correlation heatmap (Pearsons) from a dataframe. Defaults to a symmtric black-white-black scale with white being at 0 correlation.
```python
def corrheatmap(data:pd.core.frame.DataFrame, 
                vmax:float=1.0,
                diagonal:bool=False,
                decimals:int=2,
                title:str='Correlation Matrix',
                colors:List[str]=['black', 'white', 'black'],
                annot:bool = True,
                as_figure:bool=True,
                figure_params:Dict[Any,Any]={'figsize':(14,8), 'dpi':75},
                )
```

## Miscellaneous
### wcs.tools.pltfct
*Easily plot a function*

### wcs.tools.HTMLtable
*Create and modify text tables for display in Colab.* **Work In Progress Warning**

## to be continued...

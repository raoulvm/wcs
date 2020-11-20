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
### wcs.skl.confusion.pretty_confusionmatrix
*can print nicer explainable confusion matrices. pass it a confusion matrix and enjoy.* **Work In Progress Warning**

### wcs.skl.compose.get_feature_names
*returns the output columns of e.g. a column transformer with nested pipelines*

### wcs.skl.compose.repipe_transformer_tuples 
*collates transformations for the same columns into Pipelines. See DocString*

### wcs.skl.compose.make_transformer_list(tlist:list, withnames:bool=True)->list:
*instantiates transformers for multiple use of the transformation list without the need of resetting them again*

### wcs.skl.rcat
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

## Miscellaneous
### wcs.tools.pltfct
*Easily plot a function*

### wcs.tools.HTMLtable
*Create and modify text tables for display in Colab.* **Work In Progress Warning**

## to be continued...

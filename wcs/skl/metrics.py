import numpy as np
from ..tools import HTMLtable
from IPython.display import display
from typing import List, Union
from sklearn.metrics import confusion_matrix
from pandas import Series as pdSeries

def pretty_confusionmatrix( confusionmatrix: np.ndarray, 
                            textlabels:List[str]=['Positive','Negative'], 
                            title:str='Confusion Matrix', 
                            texthint:str='', 
                            metrics:bool=True, 
                            as_object:bool=False)->Union[object, dict]:
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
        order (str, optional, default = 'sklearn): The orientation of the input/output matrix. Can be 'sklearn' or 'wikipedia'
        as_object: If calculating metrics (metrics=True), return only the confusion matrix table with metrics, and no dict of metrics.

    Returns:
        Union[HTMLTable, dict]: The matrix as HTMLTable if `metrics` is set to False, a dict with the metrics otherwise (Default)
    """
    if not isinstance(confusionmatrix, np.ndarray):
        confusionmatrix = np.array(confusionmatrix)
    def mtext(text: str, hover: str = None):
        if hover is None: return '<div>'+text+'</div>'
        return f'<div title="{hover}">'+text+'</div>'
    rows, columns = confusionmatrix.shape
    m:HTMLtable = HTMLtable(2+rows,2+columns,title)
    m.merge_cells(0,0,1,1)
    population = confusionmatrix.sum()
    if texthint=='':
        m[0,0]=f'Population= {population} '
    else:
        m[0,0] = texthint
    m.merge_cells(0,2,0,1+columns)[0,2]='Predicted Class'
    m.merge_cells(2,0,1+rows,0)[2,0]='Actual Class'
    for i in range(len(textlabels)):
        if i<columns: m[1, 2+i] = textlabels[i]
        if i<rows: m[2+i, 1] = textlabels[i]
    for r in range(rows):
        for c in range(columns):
            m[2+r, 2+c] = mtext(f"<b>{confusionmatrix[r,c]}</b>", f'"{confusionmatrix[r,c]}"" predicted "{textlabels[c]}"s are "{textlabels[r]}"s')



    if metrics and rows==2 and columns==2: # metrics do not work well for other sizes
        ret_metrics = {}
        m.add_rows(4)
        m.add_columns(4)
        c1 = 2+columns
        c2 = 4+columns
        r1 = 2
        
        tpr = confusionmatrix[0,0]/confusionmatrix[0].sum()
        ret_metrics.update({'TPR':tpr})
        m[r1 +0,c1   ] = mtext('True Positive Rate = Recall = Sensitivity', f"Of all '{textlabels[0]}'s, we detected {tpr:,.0%} ")
        m[r1 +0,c1+1 ] = f'{tpr:,.0%}'

        fpr = confusionmatrix[1,0]/(confusionmatrix[1].sum())
        ret_metrics.update({'FPR':fpr})
        m[r1 +1,c1   ] = mtext('False Positive Rate = Fall-out = P(false alarm)', f"Of all '{textlabels[1]}'s, we predicted {fpr:,.0%} to be {textlabels[0]}s" ) 
        m[r1 +1,c1+1 ] = f'{confusionmatrix[1,0]/(confusionmatrix[1].sum()):,.0%}'
        
        fnr = confusionmatrix[0,1]/confusionmatrix[0].sum()
        ret_metrics.update({'FNR':fnr})
        m[r1 ,c2    ] = mtext('False Negative Rate = Miss Rate', f"Of all the '{textlabels[0]}'s, we misdetected {fnr:,.0%}" )
        m[r1 ,c2 + 1] = f'{fnr:,.0%}'

        tnr = confusionmatrix[1,1]/(confusionmatrix[1].sum())
        ret_metrics.update({'TNR':tnr})
        m[r1 +1,c2 ] = mtext('Specificity (SPC), Selectivity, True negative rate (TNR)' , f"Of all the '{textlabels[1]}'s, we correctly identified {tnr:,.0%}") 
        m[r1 +1,c2+1 ] = f'{tnr:,.0%}'



        c1 = 0
        r1 = 4
        c2 = c1+2

        # regarding population
        prevalence = (confusionmatrix[0].sum())/population
        ret_metrics.update({'prevalence':prevalence})
        m[r1    ,  c1    ] = mtext('Prevalence', f"It is so likely to hit a {textlabels[0]} randomly: {prevalence:,.1%}")
        m[r1    ,  c1 + 1] = f"{prevalence:,.1%}"
        accuracy = (confusionmatrix[0,0] + confusionmatrix[1,1])/population
        ret_metrics.update({'accuracy':accuracy})
        m[r1    ,  c2    ] = mtext('Accuracy' , f"Of all samples, we correctly identified {accuracy:,.1%}")
        m[r1    ,  c2 + 1] = f'{accuracy:,.1%}'

        # regarding predicted Positives
        m.merge_cells(r1 + 1,  c1, None,c1+1 )
        precision  = confusionmatrix[0,0] / confusionmatrix[:,0].sum()
        ret_metrics.update({'precision':precision})
        m[r1 + 1,  c1    ] = mtext('Positive Predictive Value = Precision', f"Of the predicted {textlabels[0]}s, we were right in {precision:,.0%} of the cases.")
        m[r1 + 1,  c1 + 2] = f'{precision:,.0%}'
        m.merge_cells(r1 + 2,  c1, None,c1+1 )
        fdr = confusionmatrix[1,0] / confusionmatrix[:,0].sum()
        ret_metrics.update({'FDR':fdr})
        m[r1 +2 ,  c1    ] = mtext('False Discovery Rate' , f"Of all predicted {textlabels[0]}s, we were wrong in {fdr:,.0%}")
        m[r1 +2 ,  c1 + 2] = f'{fdr:,.0%}'

        # regarding predicted Negatives
        For = confusionmatrix[0,1] / confusionmatrix[:,1].sum()
        ret_metrics.update({'FOR':For})
        m[r1 + 1,  c2 + 2 ] = mtext('False Omission Rate' , f"Of the predicted {textlabels[1]}s, {For:,.0%} were in fact {textlabels[0]}s!")
        m[r1 + 1,  c2 + 1] = f'{For:,.0%}'

        npv = confusionmatrix[1,1] / confusionmatrix[:,1].sum()
        ret_metrics.update({'NPV':npv})
        m[r1 +2 ,  c2 +2 ] = mtext('Negative predicted Value' , f"Of all predicted {textlabels[1]}s, we correctly identified {npv:,.0%}")
        m[r1 +2 ,  c2 + 1] = f'{npv:,.0%}'
        
        #m.merge_cells(r1 + 3,  c1+2, None,c1+3 )
        f1score = 2* ((confusionmatrix[0,0] / confusionmatrix[:,0].sum() *confusionmatrix[0,0]/confusionmatrix[0].sum()) ) / (confusionmatrix[0,0] / confusionmatrix[:,0].sum() + confusionmatrix[0,0]/confusionmatrix[0].sum())
        ret_metrics.update({'F1score':f1score})
        m[r1 + 3,  c1 + 2] = mtext('F1 Score')
        m[r1 + 3,  c1 + 3] = f'{f1score:,.0%}'


        # beautify only 
        #merge some cells to get rid of the clutter
        # top right
        #m.merge_cells(0, 2+columns,1, 5+columns)
        # bottom left
        #m.merge_cells(5+rows, 0, 5+rows, 1)
        # bottom right
        #m.merge_cells(2+columns, 2+rows, 5+rows, 5+columns)


        if as_object:
            return m
        display(m)
        return ret_metrics
    return m


def confusion(y_true:list, y_predict:list, labels:list='auto', textlabels:List[str]=None, title:str='Confusion Matrix'):
    """Generate a HTMLTable with confusion matrix details
       Will SORT the LABELS (if none are provided) from the data descending, assuming the Highest Value is the "Positive" label.

    Args:
        y_true (list): True Values 
        y_predict (list): Predicted Values
        labels (list, optional): The List of labels in the values, stating with the POSITIVE label!. Defaults to 'auto'.
        textlabels (List[str], optional): The decription you want to have in the confusion matrix printout, same order as in `labels`. Defaults to ['Positive', 'Negative'].
        title (str, optional): The header for the table printout. Defaults to 'Confusion Matrix'.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if len(y_true)!=len(y_predict):
        raise ValueError('Lengths are different!')
    if isinstance(labels, str) and labels=='auto':
        print(sorted(pdSeries(y_true).append(pdSeries(y_predict)).unique(), reverse=True))
        labels = sorted(pdSeries(y_true).append(pdSeries(y_predict)).unique(), reverse=True)
        print(labels)
        if textlabels is None:
            if isinstance(labels[0],str):
                textlabels = labels
    if textlabels is None: 
        textlabels = ['Positive', 'Negative']
    return pretty_confusionmatrix( confusionmatrix= confusion_matrix(y_true, y_predict, labels=labels), title=title, textlabels=textlabels, metrics=True, as_object=True)

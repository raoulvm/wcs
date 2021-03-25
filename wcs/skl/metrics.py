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
                            as_object:bool=False,
                            decimals:int=1)->Union[object, dict]:
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
        decimals (int, optional): Number of decimals for percentage values. Defaults to 1.

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
    m.merge_cells(0,2,0,1+columns)[0,2]='<b>Predicted Class</b>'
    m.merge_cells(2,0,1+rows,0)[2,0]='<b>Actual Class</b>'
    for i in range(len(textlabels)):
        if i<columns: m[1, 2+i] = '<b>'+textlabels[i] + f"</b> ({confusionmatrix[i].sum()})"
        if i<rows: m[2+i, 1] = '<b>'+textlabels[i] + f"</b> ({confusionmatrix[:,i].sum()})"
    for r in range(rows):
        for c in range(columns):
            m[2+r, 2+c] = mtext(f"<b>{confusionmatrix[r,c]}</b>", f"{confusionmatrix[r,c]} predicted {textlabels[c]}s are {textlabels[r]}s")



    if metrics and rows==2 and columns==2: # metrics do not work well for other sizes
        ret_metrics = {}
        m.add_rows(4)
        m.add_columns(4)
        c1 = 2+columns
        c2 = 4+columns
        r1 = 2

        all_positives = confusionmatrix[0].sum()
        all_negatives = confusionmatrix[1].sum()
        all_pred_positives = confusionmatrix[:,0].sum()
        all_pred_negatives = confusionmatrix[:,1].sum()
        tp = confusionmatrix[0,0]
        fp = confusionmatrix[1,0]
        fn = confusionmatrix[0,1]
        tn = confusionmatrix[1,1]

        
        tpr = tp/all_positives
        ret_metrics.update({'TPR':tpr})
        m[r1 +0,c1   ] = mtext('True Positive Rate = Recall = Sensitivity', f"Of all '{textlabels[0]}'s, we detected {tpr:,.{decimals}%} (TP/Positives) ")
        m[r1 +0,c1+1 ] = f'{tpr:,.{decimals}%}'

        fpr = fp/all_negatives
        ret_metrics.update({'FPR':fpr})
        m[r1 +1,c1   ] = mtext('False Positive Rate = Fall-out = P(false alarm)', f"Of all '{textlabels[1]}'s, we predicted {fpr:,.{decimals}%} to be {textlabels[0]}s (FP/Negatives)" ) 
        m[r1 +1,c1+1 ] = f'{confusionmatrix[1,0]/(confusionmatrix[1].sum()):,.{decimals}%}'
        
        fnr = fn/all_positives
        ret_metrics.update({'FNR':fnr})
        m[r1 ,c2    ] = mtext('False Negative Rate = Miss Rate', f"Of all the '{textlabels[0]}'s, we misdetected {fnr:,.{decimals}%} (FN/Positives)" )
        m[r1 ,c2 + 1] = f'{fnr:,.{decimals}%}'

        tnr = tn/all_negatives
        ret_metrics.update({'TNR':tnr})
        m[r1 +1,c2 ] = mtext('Specificity (SPC), Selectivity, True negative rate (TNR)' , f"Of all the '{textlabels[1]}'s, we correctly identified {tnr:,.{decimals}%} (TN/Negatives)") 
        m[r1 +1,c2+1 ] = f'{tnr:,.{decimals}%}'



        c1 = 0
        r1 = 4
        c2 = c1+2

        # regarding population
        prevalence = all_positives/population
        ret_metrics.update({'prevalence':prevalence})
        m[r1    ,  c1    ] = mtext('Prevalence', f"It is so likely to hit a {textlabels[0]} randomly: {prevalence:,.{decimals}%} (Positives/Population)")
        m[r1    ,  c1 + 1] = f"{prevalence:,.{decimals}%}"
        accuracy = (tp + tn)/population
        ret_metrics.update({'accuracy':accuracy})
        m[r1    ,  c2    ] = mtext('Accuracy' , f"Of all samples, we correctly identified {accuracy:,.{decimals}%} ((TP+TN)/Population)")
        m[r1    ,  c2 + 1] = f'{accuracy:,.{decimals}%}'

        # regarding predicted Positives
        m.merge_cells(r1 + 1,  c1, None,c1+1 )
        precision  = tp / all_pred_positives
        ret_metrics.update({'precision':precision})
        m[r1 + 1,  c1    ] = mtext('Positive Predictive Value = Precision', f"Of the predicted {textlabels[0]}s, we were right in {precision:,.{decimals}%} of the cases. (TP/Pred.Positives)")
        m[r1 + 1,  c1 + 2] = f'{precision:,.{decimals}%}'
        m.merge_cells(r1 + 2,  c1, None,c1+1 )
        fdr = fp / all_pred_positives
        ret_metrics.update({'FDR':fdr})
        m[r1 +2 ,  c1    ] = mtext('False Discovery Rate' , f"Of all predicted {textlabels[0]}s, we were wrong in {fdr:,.{decimals}%} (FP/Pred.Positives")
        m[r1 +2 ,  c1 + 2] = f'{fdr:,.{decimals}%}'

        # regarding predicted Negatives
        For = fn / all_pred_negatives
        ret_metrics.update({'FOR':For})
        m[r1 + 1,  c2 + 2 ] = mtext('False Omission Rate' , f"Of the predicted {textlabels[1]}s, {For:,.{decimals}%} were in fact {textlabels[0]}s! (FN/Pred.Negatives)")
        m[r1 + 1,  c2 + 1] = f'{For:,.{decimals}%}'

        npv = tn / all_pred_negatives
        ret_metrics.update({'NPV':npv})
        m[r1 +2 ,  c2 +2 ] = mtext('Negative predicted Value' , f"Of all predicted {textlabels[1]}s, we correctly identified {npv:,.{decimals}%} (TN/Pred.Negatives)")
        m[r1 +2 ,  c2 + 1] = f'{npv:,.{decimals}%}'
        
        #m.merge_cells(r1 + 3,  c1+2, None,c1+3 )
        f1score = 2* ((precision * tpr) ) / (precision + tpr)
        ret_metrics.update({'F1score':f1score})
        m[r1 + 3,  c1 + 2] = mtext('F1 Score', '2*(Precision*Recall)/(Precision+Recall)')
        m[r1 + 3,  c1 + 3] = f'{f1score:,.{decimals}%}'


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


def confusion(y_true:list, y_predict:list, labels:list='auto', textlabels:List[str]=None, title:str='Confusion Matrix', decimals:int=1):
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
    return pretty_confusionmatrix( 
        confusionmatrix= confusion_matrix(y_true, y_predict, labels=labels), 
        title=title, 
        textlabels=textlabels, 
        metrics=True, 
        as_object=True,
        decimals=decimals,
        )

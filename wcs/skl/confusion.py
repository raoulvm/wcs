import numpy as np
from ..tools import HTMLtable
from IPython.display import display

def pretty_confusionmatrix(confusionmatrix: np.ndarray, textlabels:list=['Positive','Negative'], title:str='Confusion Matrix', texthint:str='', metrics:bool=True)->None:
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
        
        tpp = confusionmatrix[0,0]/confusionmatrix[0].sum()
        ret_metrics.update({'TPP':tpp})
        m[r1 +0,c1   ] = mtext('True Positive Rate = Recall = Sensitivity', f"Of all '{textlabels[0]}'s, we detected {tpp:,.0%} ")
        m[r1 +0,c1+1 ] = f'{tpp:,.0%}'

        fpr = confusionmatrix[1,0]/(confusionmatrix[1].sum())
        ret_metrics.update({'FPP':fpr})
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
        fdr = confusionmatrix[0,1] / confusionmatrix[:,0].sum()
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
        
        m.merge_cells(r1 + 3,  c1+2, None,c1+3 )
        f1score = 2* ((confusionmatrix[0,0] / confusionmatrix[:,0].sum() *confusionmatrix[0,0]/confusionmatrix[0].sum()) ) / (confusionmatrix[0,0] / confusionmatrix[:,0].sum() + confusionmatrix[0,0]/confusionmatrix[0].sum())
        ret_metrics.update({'F1score':f1score})
        m[r1 + 3,  c1 + 2] = mtext('F1 Score')
        m[r1 + 3,  c1 + 4] = f'{f1score:,.0%}'

        #m.mergeCells(5,4,7,7)
        display(m)
        ret_metrics
    return m


import sys
import numpy
import scipy.stats
from seaborn import heatmap

if 'diff_predictor.core' not in sys.modules:
    from diff_predictor import core


def perf_meas(y_actual, y_pred, cls, verbose=True):
    '''
    Shows the performance measurements of resulting prediction.
    Performance measures include true positive, true negative,
    false positive, false negative
    Parameters
    ----------
    y_actual : list
        Actual values of y
    y_pred : list
        Predicted values of y
    cls : int
        class to run performance measure on
    verbose : boolean : True
        report performance as a string
    Returns
    -------
    tuple of four performance values (TP, FP, TN, FN)
    '''

    assert len(y_actual) == len(y_pred), 'Must be same number of actual and predicted values'

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_actual)): 
        if (y_actual[i]==y_pred[i]) and (y_pred[i]==cls):
           TP += 1
        if (y_pred[i]==cls) and (y_actual[i]!=y_pred[i]):
           FP += 1
        if (y_actual[i]==y_pred[i]) and (y_pred[i]!=cls):
           TN += 1
        if (y_pred[i]!=cls) and (y_actual[i]!=y_pred[i]):
           FN += 1
    if verbose is True:
        print(f'(TP, FP, TN, FN) = {(TP, FP, TN, FN)}')
    return(TP, FP, TN, FN)


def corrmat(df, method='pearson', show_plot=True, **kwargs):
    '''
    
    '''
    plot_options = {'annot': True, 
                    'fmt': "f",
                    }
    plot_options.update(kwargs)
    error_msg = "Correlation type not available. Select" +\
                "from pearson, spearman, or kendall corr."
    switch_case = {'pearson': df.corr(),
                   'spearman': df.corr(method=method),
                   'kendall': df.corr(method=method)}
    corr_mat = switch_case.get(method, lambda: error_msg)
    if show_plot:
        return heatmap(corr_mat, **plot_options)
    return corr_mat
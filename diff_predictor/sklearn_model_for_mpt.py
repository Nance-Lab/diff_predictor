import sys
import operator
import pandas as pd
import numpy as np
import xgboost as xgb
import opendataval as odv
import torch

from xgboost import callback, DMatrix, Booster
from xgboost.callback import EarlyStopping, EvaluationMonitor

from xgboost.core import CallbackEnv, EarlyStopException, STRING_TYPES # TODO update this
from xgboost.training import CVPack

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import Cls

from opendataval.model import ClassifierSkLearnWrapper
# from odv.dataloader import DataFetcher # fetcher has covar/label dim information
# from odv.model import ModelFactory

'''
Wrapper module that allows a user to choose any model architecture
listed on scikit-learn, and train it on MPT data. Aims to ensure that,
regardless of what model is used, hyperparameter tuning and train test
validation splitting appropriately have no leakage of trajectories that
are in the same microenvironment.
TODO: wait -- should this wrapper also accommodate non-sklearn models like xgboost?
      or should we keep predxgboost separate? we still need to update predxgboost.py
'''

def paramsearch(X_train, y_train, features, init_params, nfold=5,
                    early_stopping_rounds=3, **kwargs):
    '''
    Extensive parameter search for a model listed on scikit-learn.
    Uses random search to tune hyperparameters to a .

    Parameters
    ----------
    X_train : pandas.DataFrame
        X data to be trained
    y_train : pandas.DataFrame
        y data to be trained
    features : list
        features selected to be trained
    init_params : dict
        initial parameters for model. This will be used in the initial
        calculation for evaluation and will be compared to the next params
        in grid search
    nfold : int : 5
        Number of folds for cross-validation
    early_stopping_rounds : int : 3
        Number of rounds allowed to accept converge

    Optional Parameters
    -------------------
    (xgboost) num_boost_round : int
        Maximum number of boosted decsion tree rounds to run through
    use_gpu : boolean : False
        Use cuda processing if gpu supports it
    metrics : list
        Metrics to track
    early_break : int : 5
        maximum number of times the random gridsearch while difference in
        starting and ending evaluation value is within the given threshold
    thresh : float : 0.01
        allowed threshold between difference in starting and ending evaluation
        value
    seed : int : 1111
        random seed to start on.
    gs_params : dict
        extra parameters to use in gridsearch. Note: function will not optimize
        these parameters

    Returns
    -------
    best_model : dataframe
        results of best crossvalidated model with metrics of each boosted round
    best_param : dict
        hyperparameters of best found model
    best_eval : float
        resulting evaluation metric of best boosted round metric
    (xgboost) best_boost_rounds : int, or None if num_boost_round unspecified
        number of boost rounds to converge
    '''
    params = {**init_params}

    if 'use_gpu' in kwargs and kwargs['use_gpu']:
        # GPU integration will cut cv time in ~half:
        params.update({'gpu_id': 0,
                        'tree_method': 'gpu_hist',
                        'predictor': 'gpu_predictor'})

    if 'metrics' not in kwargs:
        metrics = {params['eval_metric']}
    else:
        metrics = kwargs['metrics']
        metrics.append(params['eval_metric'])

    if params['eval_metric'] in ['map', 'auc', 'aucpr']:
        eval_f = operator.gt
    else:
        eval_f = operator.lt

    early_break = kwargs.get('early_break', 5)
    thresh = kwargs.get('thresh', 0.01)
    seed = kwargs.get('seed', 1111)

    gs_params = {'subsample': np.random.choice([i/10. for i in range(5, 11)], 3),
                'colsample': np.random.choice([i/10. for i in range(5, 11)], 3),
                'eta': np.random.choice([.005, .01, .05, .1, .2, .3], 3),
                'gamma': [0] + list(np.random.choice([0.01, 0.001, 0.2, 0.5, 1.0,
                                                        2.0, 3.0, 5.0, 10.0], 3)),
                'max_depth': [10] + list(np.random.randint(1, 10, 3)),
                'min_child_weight': [0, 10] + list(np.random.randint(0, 10, 3))}

    if 'gs_params' in kwargs:
        gs_params.update(kwargs['gs_params'])

    best_param = params

    num_boost_round = kwargs.get('num_boost_round', None)

    if num_boost_round: # xgboost
        best_model = cv(params,
                      X_train,
                      y_train,
                      features,
                      nfold=nfold,
                      early_stopping_rounds=early_stopping_rounds,
                      metrics=metrics,
                      num_boost_round=num_boost_round)
        best_boost_rounds = best_model[f"test-{params['eval_metric']}-mean"].idxmin()
    else: # rf
        best_model = cv(params,
                      X_train,
                      y_train,
                      features,
                      nfold=nfold,
                      early_stopping_rounds=early_stopping_rounds,
                      metrics=metrics)

    best_eval = best_model[f"test-{params['eval_metric']}-mean"].min()

    def _gs_helper(var1n, var2n, best_model, best_param, best_eval, **kwargs):
        '''
        Helper function for paramsearch.
        '''
        local_param = {**best_param}
        for var1, var2 in gs_param:
            print(f"Using CV with {var1n}={{{var1}}}, {var2n}={{{var2}}}")
            local_param[var1n] = var1
            local_param[var2n] = var2
            cv_model = cv(local_param,
                          X_train,
                          y_train,
                          features,
                          nfold=nfold,
                          early_stopping_rounds=early_stopping_rounds,
                          metrics=metrics,
                          **kwargs)
            cv_eval = cv_model[f"test-{local_param['eval_metric']}-mean"].min()
            if 'best_boost_rounds' in kwargs: #xgboost
                boost_rounds = cv_model[f"test-{local_param['eval_metric']}-mean"].idxmin()
                if (eval_f(cv_eval, best_eval)):
                    best_model = cv_model
                    best_param[var1n] = var1
                    best_param[var2n] = var2
                    best_eval = cv_eval
                    if not best_boost_rounds:
                        best_boost_rounds = boost_rounds # only update if not None
                    print(f"New best param found: "
                          f"{local_param['eval_metric']} = {{{best_eval}}}, "
                          f"boost_rounds = {{{best_boost_rounds}}}")
            else: #rf
                if (eval_f(cv_eval, best_eval)):
                    best_model = cv_model
                    best_param[var1n] = var1
                    best_param[var2n] = var2
                    best_eval = cv_eval
                    print(f"New best param found: "
                          f"{local_param['eval_metric']} = {{{best_eval}}}")
        if 'best_boost_rounds' in kwargs:
            return (best_model, best_param, best_eval, best_boost_rounds)
        else:
            return (best_model, best_param, best_eval)

    while early_break > 0:
        np.random.seed(seed)
        best_eval_init = best_eval
        gs_param = {
            (subsample, colsample)
            for subsample in gs_params['subsample']
            for colsample in gs_params['colsample']
        }
        if best_boost_rounds:
            best_model, best_param, best_eval, best_boost_rounds = _gs_helper('subsample',
                                                                              'colsample_bytree',
                                                                              best_model,
                                                                              best_param,
                                                                              best_eval,
                                                                              best_boost_rounds)
        else:
            best_model, best_param, best_eval = _gs_helper('subsample',
                                                            'colsample_bytree',
                                                            best_model,
                                                            best_param,
                                                            best_eval)
        gs_param = {
            (max_depth, min_child_weight)
            for max_depth in gs_params['max_depth']
            for min_child_weight in gs_params['min_child_weight']
        }
        if best_boost_rounds: #xgboost
            best_model, best_param, best_eval, best_boost_rounds = _gs_helper('max_depth',
                                                                              'min_child_weight',
                                                                              best_model,
                                                                              best_param,
                                                                              best_eval,
                                                                              best_boost_rounds)
        else: #rf
            best_model, best_param, best_eval = _gs_helper('max_depth',
                                                           'min_child_weight',
                                                            best_model,
                                                            best_param,
                                                            best_eval)
        gs_param = {
            (eta, gamma)
            for eta in gs_params['eta']
            for gamma in gs_params['gamma']
        }
        if best_boost_rounds: #xgboost
            best_model, best_param, best_eval, best_boost_rounds = _gs_helper('eta',
                                                                              'gamma',
                                                                              best_model,
                                                                              best_param,
                                                                              best_eval,
                                                                              best_boost_rounds)
        else: #rf
            best_model, best_param, best_eval = _gs_helper('eta',
                                                           'gamma',
                                                            best_model,
                                                            best_param,
                                                            best_eval)
        if (abs(best_eval_init - best_eval) < thresh):
            early_break -= 1
        seed += 1

    if num_boost_round: # xgboost
        return (best_model, best_param, best_eval, best_boost_rounds)
    else:
        return (best_model, best_param, best_eval)


def train(model, param, dtrain, dtest, dval=None, evals=None, num_round=2000, verbose=True):
    '''
    Parameters
    ----------
    model:
        A model (from sklearn) to be trained and returned.
    param : dict
        dictionary of parameters used for training.
        max_depth : maximum allowed depth of a tree,
        eta : step size shrinkage used toprevent overfiiting,
        min_child_weight : minimum sum of instance weight (hessian)
                           needed in a child,
        verbosity : verbosity of prited messages,
        objective : learning objective,
        num_class : number of classes in prediction,
        gamma : minimum loss reduction required to make a further
            partition on a leaf node of the tree,
        subsample : subsample ratio of the training instances,
        colsample_bytree : subsample ratio of columns when
                           constructing each tree,
        eval_metric : eval metric used for validatiion data
    dtrain : xgboost.DMatrix TODO dtrain: pandas.DataFrame -- should this be split up into x_train and y_train?
        training data for fitting.
    dtest : xgboost.DMatrix TODO alt: pandas.DataFrame
        testing data for fitting.

    Optional Parameters
    -------------------
    dval : xgboost.DMatrix : None TODO I feel like this is xgboost specific?
        optional evaluation data for fitting.
    evals : list : None TODO I feel like this is xgboost specific?
        evaluation configuration. Will report results in this form. If dval is
        used, will automatically update to [(dtrain, `train`), (dval, `eval`)].
        Will use the last evaulation value in the list to test for loss
        convergence
    (xgboost) num_rounds : int : 2000
        Number of boosting rounds to go through when training. A higher number
        makes a more complex ensemble model.
    (xgboost) num_boost_round : int : None
        Number of boosting iterations.

    Returns
    -------
    model : xgboost.Classifier TODO - generalize
        Resulting trained model.
    acc : float
        Accuracy of trained model.
    label: TODO - what type is this?
    preds:
        Predictions from trained model.
    '''
    if evals is None:
        evals = [(dtrain, 'train')]
    if dval is not None and (dval, 'eval') not in evals:
        evals += [(dval, 'eval')]

    trained_model, ypred = None, None

    if type(model) == xgb.Booster or type(model) == xgb.XGBClassifier:
        # model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        # TODO - might want to use this instead to conform to sklearn
        trained_model = xgb.train(param, dtrain, num_round, evals, verbose_eval=verbose)
        ypred = trained_model.predict(dtest)
    else:
        num_classes = param['num_class']
        wrapped_model = ClassifierSkLearnWrapper(model, num_classes)
        trained_model = wrapped_model.fit()
        # fit(x_train: Tensor | Dataset, y_train: Tensor | Dataset, *args, sample_weight: Tensor | None = None, **kwargs)
        # https://opendataval.github.io/opendataval.model.html#opendataval.model.api.ClassifierSkLearnWrapper.fit
        dtest_tensor = torch.tensor(dtest.values)
        ypred = wrapped_model.predict(dtest_tensor)
        # fit(x_train: Tensor | Dataset, y_train: Tensor | Dataset, *args, sample_weight: Tensor | None = None, **kwargs)
        # https://opendataval.github.io/opendataval.model.html#opendataval.model.api.ClassifierSkLearnWrapper.predict

    true_label = dtest.get_label()
    preds = [np.where(x == np.max(x))[0][0] for x in ypred]
    acc = accuracy_score(true_label, preds)
    print("Accuracy:", acc)
    return trained_model, acc, true_label, preds


#########################################################################################
######### HELPER/INTERNAL FUNCTIONS
#########################################################################################

def cv(params, X_train, y_train, features=None, nfold=3,
       folds=None, metrics=(), obj=None, feval=None,
       maximize=False, early_stopping_rounds=3,
       as_pandas=True, verbose_eval=None, show_stdv=True, seed=1234,
       callbacks=None, **kwargs):
    '''
    Cross-validation with given parameters. Modified to use spatial data with
    statistical features without risking data bleed.

    Parameters
    ----------
    params : dict
        Model params
    X_train : pandas.DataFrame
        X data to be trained
    y_train : pandas.DataFrame
        y data to be trained
    features : list
        features selected to be trained
    nfold : int : 3
        Number of folds in CV.
    folds : a KFold or StratifiedKFold instance or list of fold indeces
        Sklearn KFolds or StratifiedKFolds object.
        Alternatively may explicitly pass sample indices for each fold.
        For ``n`` folds, **folds** should be a length ``n`` list of tuples.
        Each tuple is ``(in,out)`` where ``in`` is a list of indices to be used
        as the training samples for the ``n``th fold and ``out`` is a list of
        indices to be used as the testing samples for the ``n``th fold.
    metrics : string ot list of strings
        Evaluation metrics to be watches in CV.
    obj : function
        Custom objective function.
    feval : function
        Custom evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds : int
        Activates early stopping. Cross-validation metric (average of
        validation metric computed over CV folds) needs to improve at least
        once in every **early_stopping_rounds** round(s) to continue training.
        The last entry in the evaluation history will represent the best
        iteration. If there's more than one metric in the **eval_metric**
        parameter given **params**, the last metric will be used for early
        stopping.
    fpreproc : function
        TODO: how to deal with dtrain, dtest? ignore for now
        Preprocessing function that takes (dtrain, dtest, param) and returns
        transformed versions of those.
    as_pandas : bool : True
        Return pd.DataFrame when pandas is installed.
        If False or pandas is not installed, return np.ndarray
    verbose_eval : bool, int, or None : None
        Whether to display the progress. If None, progress will be displayed
        when np.ndarray is returned. If True, progress will be displayed at
        boosting stage. If an integer is given, progress will be displayed at
        every given `verbose_eval` boosting stage.
    show_stdv : bool : True
        Whether to display the standard deviation in progress.
        Results are not affected, and always contains std.
    seed : int : 1234
        seed used to generate folds (passed to numpy.random.seed).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks using :ref:`Callback API
        <callback_api>`.
        Example:
            .. code-block:: python
            [xgb.callback.reset_learning_rate(custom_rates)]

    Optional Parameters
    -------------------
    (xgboost) num_boost_round : int : None
        Number of boosting iterations.

    Returns
    -------
    results : pandas.DataFrame
        Results of cross-validated model.
    '''
    metrics = list(metrics)

    if isinstance(params, list):
        _metrics = [x[1] for x in params if x[0] == 'eval_metric']
        params = dict(params)
        if 'eval_metric' in params:
            params['eval_metric'] = _metrics
    else:
        params = dict((k, v) for k, v in params.items())

    if (not metrics) and 'eval_metric' in params:
        if isinstance(params['eval_metric'], list):
            metrics = params['eval_metric']
        else:
            metrics = [params['eval_metric']]

    params.pop("eval_metric", None)
    results = {}

    # create folds in data
    cvfolds, wt_list = mknfold(X_train, y_train, nfold, metrics, features, params)

    # set up callbacks
    callbacks = [] if callbacks is None else callbacks
    if early_stopping_rounds is not None:
        callbacks.append(callback.EarlyStopping(early_stopping_rounds, maximize=maximize))
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.callback.EarlyStopping
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.EvaluationMonitor(show_stdv=show_stdv))
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.callback.EvaluationMonitor
        # TODO: what to do about show_stdv? it's not a param in the newest version
    # elif isinstance(verbose_eval, int): TODO: there is no verbose option in the newest version, should we create our own?
    #     callbacks.append(callback.EvaluationMonitor(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = [cb for cb in callbacks if cb.__dict__.get('before_iteration', False)]
    callbacks_after_iter = [cb for cb in callbacks if not cb.__dict__.get('before_iteration', False)]

    '''
    TODO:
    scikit learn ways to do this, other packages?
    how xgboost is currently doing it for cv; look into xgboost for callbacks
    package: optuna: hyperparam optimization--use by next stage of package (effective for DL);
    could be worth trying to use this; works with scikit learn, rf, etc.
    capable of doing what we want to do, applicable for various model types
    optuna has own system for callback tracking (more up to date)
    callbacks: for ML iterating
    '''
    num_boost_round = kwargs.get('num_boost_round', None)
    if num_boost_round:
        for i in range(num_boost_round):
            for cb in callbacks_before_iter:
                cb(CallbackEnv(model=None,
                               cvfolds=cvfolds,
                               iteration=i,
                               begin_iteration=0,
                               end_iteration=num_boost_round,
                               rank=0,
                               evaluation_result_list=None))
            for fold in cvfolds:
                fold.update(i, obj)
            res = aggcv([f.eval(i, feval) for f in cvfolds], wt_list)
            for key, mean, std in res:
                if key + '-mean' not in results:
                    results[key + '-mean'] = []
                if key + '-std' not in results:
                    results[key + '-std'] = []
                results[key + '-mean'].append(mean)
                results[key + '-std'].append(std)
            try:
                for cb in callbacks_after_iter:
                    cb(CallbackEnv(model=None,
                                   cvfolds=cvfolds,
                                   iteration=i,
                                   begin_iteration=0,
                                   end_iteration=num_boost_round,
                                   rank=0,
                                   evaluation_result_list=res))
            except EarlyStopException as e:
                for k in results:
                    results[k] = results[k][:(e.best_iteration + 1)]
                break
    else:
        # TODO: what to do for RandomForestClassifier?
        # use generalizable callback
        pass

    if as_pandas:
        results = pd.DataFrame.from_dict(results)

    return results


def aggcv(rlist, wt_list):
    '''
    Aggregate cross-validation results. Modified to use spatial
    data with statistical features without risking data bleed.

    Parameters
    ----------
    rlist : list
        list of results from each cross-validation fold
    wt_list : list
        list of weights for each fold to apply to result
    Returns
    -------
    results : list
        list of weighted results based on the desired metric. Will output
        a list of tuple containing the result, mean, and stdev.
    '''
    cvmap = {}
    idx = rlist[0].split()[0]
    for line in rlist:
        arr = line.split()
        assert idx == arr[0]
        for metric_idx, it in enumerate(arr[1:]):
            if not isinstance(it, STRING_TYPES):
                it = it.decode()
            k, v = it.split(':')
            if (metric_idx, k) not in cvmap:
                cvmap[(metric_idx, k)] = []
            cvmap[(metric_idx, k)].append(float(v))
    msg = idx
    results = []
    for (metric_idx, k), v in sorted(cvmap.items(), key=lambda x: x[0][0]):
        v = np.array(v)
        if not isinstance(msg, STRING_TYPES):
            msg = msg.decode()
        mean = np.average(v, weights=wt_list)
        std = np.sqrt(np.average((v-mean)**2, weights=wt_list))
        results.extend([(k, mean, std)])
    return results


def mknfold(X_train, y_train, nfold, evals=(), features=None, **kwargs):
    '''
    Makes n folds in input data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        X data to be trained
    y_train : pandas.DataFrame
        y data to be trained
    nfold : int
        Number of folds in CV.
    evals : list
        Evaluation metrics to be watches in CV.
    features : list
        features selected to be trained

    Optional Parameters
    -------------------
    param : dict
        model params

    Returns
    -------
    ret : list
       TODO: CVPack irrelevant to RFClassifier?
          yes -- scikit learn how they handle cv, split st there is no spatial bias
          OR could have our own CVPack
          they wrote specific one bc of tensor
        list of CVPack objects containing the dmatrix training, testing, and
        list of parameters and metrics to use for every fold
    wt_list : list
        list of weights for each fold. This is the size of each fold
    '''

    def bin_fold(X_train, nfold):
        '''
        Bins data into their respective folds.
        Parameters
        ----------
        X_train : pandas.DataFrame
            X data to be trained
        nfold : int
            number of folds desired in cv
        Returns
        -------
        bin_list : list
            list of list of indeces for each fold. For every fold in data,
            will have list of indeces which will be used as testing data
        wt_list : list
            list of weights for each fold. This is the size of each fold
        '''
        bin_list = [X_train[X_train['bins'] == i_bin].index.to_numpy() for
                    i_bin in X_train.bins.unique()]
        bin_list = sorted(bin_list, key=len)
        i = 0
        while (len(bin_list) > nfold):
            if (i >= len(bin_list)-1):
                i = 0
            bin_list[i] = np.concatenate([bin_list[i], bin_list.pop()])
            i += 1
        wt_list = [len(i)/sum(len(s) for s in bin_list) for i in bin_list]
        return bin_list, wt_list

    out_idset, wt_list = bin_fold(X_train, nfold)
    in_idset = [np.concatenate([out_idset[i]
                                for i in range(nfold) if k != i])
                for k in range(nfold)]
    evals = list(evals)
    ret = []
    for k in range(nfold):
        # perform the slicing using the indexes determined by the above methods
        x_train_snip = X_train.loc[in_idset[k]][features]
        y_train_snip = X_train.loc[in_idset[k]]['encoded_target']
        x_test_snip = X_train.loc[out_idset[k]][features]
        y_test_snip = X_train.loc[out_idset[k]]['encoded_target']

        param = kwargs.get('param', None)
        if param:
          dtrain = DMatrix(x_train_snip, label=y_train_snip)
          dtest = DMatrix(x_test_snip, label=y_test_snip)
          tparam = param
          plst = list(tparam.items()) + [('eval_metric', itm) for itm in evals]
          ret.append(CVPack(dtrain, dtest, plst))
        # TODO: figure out what to do in other case
        # DMatrix -- package features and target, doing training and testing and calc metrics
        # has physical features and true target
        # one object but two diff things
        # scikit learn prob use dfs or arrays
    return ret, wt_list


def pandas_df_to_dmatrix(df, label):
    '''
    (XGBoost-specific)
    Converts a pandas dataframe to a DMatrix.

    Parameters
    ----------
    df: pandas DataFrame
        The dataframe to convert.
    label:
        The label to use.

    Returns
    -------
    dmatrix: xgboost.DMatrix
        The resulting DMatrix.
    '''
    return DMatrix(df, label=label)


def pandas_df_to_pytorch_tensor(df: pd.DataFrame):
    '''
    (XGBoost-specific)
    Converts a pandas DataFrame to a PyTorch tensor.

    Parameters
    ----------
    df: pandas DataFrame
        The dataframe to convert.

    Returns
    -------
    torch: PyTorch.torch
        The resulting Pytorch tensor.
    '''
    return torch.tensor(df.values)


class XGBoostCallback(xgb.callback.TrainingCallback):
    '''
    To replace CallbackEnv? TODO - reimplement this?
    '''
    def __init__(self):
        pass
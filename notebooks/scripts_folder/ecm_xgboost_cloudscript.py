from azureml.core import Run
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, getcwd, chdir
from os.path import isfile, join
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import operator
import xgboost as xgb
import shap
from xgboost.training import CVPack
from xgboost import callback
from xgboost.core import CallbackEnv
from xgboost.core import EarlyStopException
from xgboost.core import STRING_TYPES

# Get the experiment run context
run = Run.get_context()

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
    while(len(bin_list) > nfold):
        if (i >= len(bin_list)-1):
            i = 0
        bin_list[i] = np.concatenate([bin_list[i], bin_list.pop()])
        i += 1
    wt_list = [len(i)/sum(len(s) for s in bin_list) for i in bin_list]
    return bin_list, wt_list


def mknfold(X_train, y_train, nfold, param, evals=(), features=None):
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
    param : dict
        Booster params
    evals : list
        Evaluation metrics to be watches in CV.
    features : list
        features selected to be trained
        
    Returns
    -------
    ret : list
        list of CVPack objects containing the dmatrix training, testing, and
        list of parameters and metrics to use for every fold
    wt_list : list
        list of weights for each fold. This is the size of each fold
    '''
    #if not features:
        #features = X_train.columns
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
        dtrain = xgb.DMatrix(x_train_snip, label=y_train_snip)
        dtest = xgb.DMatrix(x_test_snip, label=y_test_snip)
        tparam = param
        plst = list(tparam.items()) + [('eval_metric', itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret, wt_list


def cv(params, X_train, y_train, features=None, num_boost_round=20, nfold=3,
       folds=None, metrics=(), obj=None, feval=None,
       maximize=False, early_stopping_rounds=None, fpreproc=None,
       as_pandas=True, verbose_eval=None, show_stdv=True, seed=1234,
       callbacks=None):
    '''
    Cross-validation with given parameters. Madified from cv method found in
    xgboost package (https://github.com/dmlc/xgboost) to use spatial data with
    statistical features without risking data bleed.

    Parameters
    ----------
    params : dict
        Booster params
    X_train : pandas.DataFrame
        X data to be trained
    y_train : pandas.DataFrame
        y data to be trained
    features : list
        features selected to be trained
    num_boost_round : int : 20
        Number of boosting iterations.
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
    
    
    Returns
    -------
    evaluation history : list(String)
    '''
    if isinstance(metrics, str):
        metrics = [metrics]
    #if not features:
        #features = X_train.columns
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
    cvfolds, wt_list = mknfold(X_train,
                               y_train,
                               nfold,
                               params,
                               metrics,
                               features)
    # setup callbacks
    callbacks = [] if callbacks is None else callbacks
    if early_stopping_rounds is not None:
        callbacks.append(callback.early_stop(early_stopping_rounds,
                                             maximize=maximize,
                                             verbose=False))
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, int):
        callbacks.append(callback.print_evaluation(verbose_eval,
                                                   show_stdv=show_stdv))
    callbacks_before_iter = [
        cb for cb in callbacks if
        cb.__dict__.get('before_iteration', False)]
    callbacks_after_iter = [
        cb for cb in callbacks if not
        cb.__dict__.get('before_iteration', False)]
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
    if as_pandas:
        results = pd.DataFrame.from_dict(results)
    return results


def aggcv(rlist, wt_list):
    # pylint: disable=invalid-name
    '''
    Aggregate cross-validation results. Madified from cv method found
    in xgboost package (https://github.com/dmlc/xgboost) to use spatial
    data with statistical features without risking data bleed.

    If verbose_eval is true, progress is displayed in every
    call. If verbose_eval is an integer, progress will only be
    displayed every `verbose_eval` trees, tracked via trial.
    
    Parameters
    ----------
    rlist : 
    wt_list : 
    Returns
    -------
    results : 
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
        std = np.average((v-mean)**2, weights=wt_list)
        results.extend([(k, mean, std)])
    return results


def xgb_paramsearch(X_train, y_train, features, init_params, nfold=5, 
                    num_boost_round=2000, early_stopping_rounds=3, 
                    **kwargs):
    '''
    Makes n folds in input data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        X data to be trained
    y_train : pandas.DataFrame
        y data to be trained
    features : list
        features selected to be trained
    init_params : dict
        initial parameters for Booster. This will be used in the initial
        calculation for evaluation and will be compared to the next params
        in grid search
    nfold : int : 5
        Number of folds
    num_boost_round : int : 2000
    early_stopping_rounds : int : 3
    
    Optional Parameters
    -------------------
    use_gpu : boolean : False
        Use cuda processing if gpu supports it
    metrics : list
        xgboost metrics to track
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
    best_model : 
    best_param : 
    best_eval : 
    best_boost_rounds : 
    '''
    params = {**init_params}
    if 'use_gpu' in kwargs and kwargs['use_gpu']:
        # GPU integration will cut cv time in ~half:
        params.update({'gpu_id' : 0,
                       'tree_method': 'gpu_hist',
                       'predictor': 'gpu_predictor'})
    if 'metrics' not in kwargs:
        metrics = {params['eval_metric']}
    else:
        metrics.add(params['eval_metric'])
    if params['eval_metric'] in ['map', 'auc', 'aucpr']:
        eval_f = operator.gt
    else: 
        eval_f = operator.lt
    if 'early_break' not in kwargs:
        early_break = 5
    else: 
        early_break = kwargs['early_break']
    if 'thresh' not in kwargs:
        thresh = 0.01
    else: 
        thresh = kwargs['thresh']
    if 'seed' not in kwargs:
        seed = 1111
    else: 
        seed = kwargs['seed']
    gs_params = {
        'subsample': np.random.choice([i/10. for i in range(5,11)], 3),
        'colsample': np.random.choice([i/10. for i in range(5,11)], 3),
        'eta': np.random.choice([.005, .01, .05, .1, .2, .3], 3),
        'gamma': [0] + list(np.random.choice([0.01, 0.001, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0], 3)),
        'max_depth': [10] + list(np.random.randint(1, 10, 3)),
        'min_child_weight': [0, 10] + list(np.random.randint(0, 10, 3))
        }
    if 'gs_params' in kwargs:
        gs_params.update(kwargs['gs_params'])
    best_param = params
    best_model = cv(params, 
                    X_train, 
                    y_train, 
                    features, 
                    nfold=nfold, 
                    num_boost_round=num_boost_round, 
                    early_stopping_rounds=early_stopping_rounds, 
                    metrics=metrics)
    best_eval = best_model[f"test-{params['eval_metric']}-mean"].min()
    best_boost_rounds = best_model[f"test-{params['eval_metric']}-mean"].idxmin()
    def _gs_helper(var1n, var2n, best_model, best_param,
                   best_eval, best_boost_rounds):
        '''
        Helper function for xgb_paramsearch. 
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
                          num_boost_round= num_boost_round, 
                          early_stopping_rounds=early_stopping_rounds, 
                          metrics=metrics)
            cv_eval = cv_model[f"test-{local_param['eval_metric']}-mean"].min()
            boost_rounds = cv_model[f"test-{local_param['eval_metric']}-mean"].idxmin()
            if(eval_f(cv_eval, best_eval)):
                best_model = cv_model
                best_param[var1n] = var1
                best_param[var2n] = var2
                best_eval = cv_eval
                best_boost_rounds = boost_rounds
                print(f"New best param found: "
                      f"{local_param['eval_metric']} = {{{best_eval}}}, "
                      f"boost_rounds = {{{best_boost_rounds}}}")
        return best_model, best_param, best_eval, best_boost_rounds
    while(early_break >= 0):
        np.random.seed(seed)
        best_eval_init = best_eval
        gs_param = {
            (subsample, colsample)
            for subsample in gs_params['subsample']
            for colsample in gs_params['colsample']
        }
        best_model,
        best_param,
        best_eval,
        best_boost_rounds = _gs_helper('subsample', 
                                       'colsample_bytree', 
                                       best_model, 
                                       best_param, 
                                       best_eval, 
                                       best_boost_rounds)
        gs_param = {
            (max_depth, min_child_weight)
            for max_depth in gs_params['max_depth']
            for min_child_weight in gs_params['min_child_weight']
        }
        best_model,
        best_param,
        best_eval,
        best_boost_rounds = _gs_helper('max_depth', 
                                       'min_child_weight', 
                                       best_model, 
                                       best_param, 
                                       best_eval, 
                                       best_boost_rounds)
        gs_param = {
            (eta, gamma)
            for eta in gs_params['eta']
            for gamma in gs_params['gamma']
        }
        best_model,
        best_param,
        best_eval,
        best_boost_rounds = _gs_helper('eta', 
                                       'gamma', 
                                       best_model, 
                                       best_param, 
                                       best_eval, 
                                       best_boost_rounds)
        if (abs(best_eval_init - best_eval) < thresh):
            early_break-=1
        seed+=1
    return best_model, best_param, best_eval, best_boost_rounds


def train(param, dtrain, dtest, dval=None, evals=None, num_round=2000):
    '''
    Parameters
    ----------
    param : dict
        dictionary of parameters used for training.
        max_depth : maximum allowed depth of a tree,
        eta : step size shrinkage used toprevent overfiiting,
        min_child_weight : minimum sum of instance weight (hessian) needed in a child,
        verbosity : verbosity of prited messages,
        objective : learning objective,
        num_class : number of classes in prediction,
        gamma : minimum loss reduction required to make a further
            partition on a leaf node of the tree,
        subsample : subsample ratio of the training instances,
        colsample_bytree : subsample ratio of columns when constructing each tree,
        eval_metric : eval metric used for validatiion data
        (https://xgboost.readthedocs.io/en/latest/parameter.html)
    dtrain : xgb.DMatrix
        training data for fittting.
    dtest : xgb.DMatrix
        testing data for fitting.
    dval : xgb.DMatrix : None
        optional evaluation data for fitting.
    evals : list : [(dtrain, `train`)]
        evaluation configuration. Will report results in this form. If dval is
        used, will automatically update to [(dtrain, `train`), (dval, `eval`)].
        Will use the last evaulation value in the list to test for loss convergence
    num_rounds : int : 2000
        Number of boosting rounds to go through when training. A higher number
        makes a more complex ensemble model
    Returns
    -------
    model : xgb.Classifier
        Resulting trained model.
    acc : float
        accuracy of trained model
    '''
    if dval is not None and (dval, 'eval') not in evals:
        evals += [(dval, 'eval')]
    model = xgb.train(param, dtrain, num_round, evals, )
    true_label = dtest.get_label()
    ypred = model.predict(dtest)
    preds = [np.where(x == np.max(x))[0][0] for x in ypred]
    acc = metrics.accuracy_score(true_label, preds)
    print("Accuracy:",acc)
    return model, acc

def generate_fullstats(dataset_path, filelist, targets, target_col_name='Target'):
    """
    Generates single csv of all statatistics from list of files
    Parameters
    ---------
    dataset_path: string
        string of path to folder containing data files
    filelist: list
        list containing filenames of all files to be processed
    targets: list
        list containing strings that state which class/group a file is from,
        string must be in the filename of the data files
    Target: string
        
    Returns
    -------
    fstats_tot: pandas.DataFrame
        dataframe containing all rows from data files and with new column
        for the class/group the row came from
    """
    fstats_tot = None
    video_num = 0
    for filename in filelist:
            fstats = pd.read_csv(dataset_path + filename, encoding = "ISO-8859-1", index_col='Unnamed: 0')
            #print('{} size: {}'.format(filename, fstats.shape))
            
            for i in range(0, len(targets)):
                print(targets[i])
                if targets[i] in filename:
                    print('Adding file {} size: {}'.format(filename, fstats.shape))
                    fstats[target_col_name] = pd.Series(fstats.shape[0]*[targets[i]], index=fstats.index)

                    fstats['Video Number'] = pd.Series(fstats.shape[0]*[video_num], index=fstats.index)
                    if fstats_tot is None:
                        fstats_tot = fstats
                    else:
                        fstats_tot = fstats_tot.append(fstats, ignore_index=True)
                    video_num += 1
                    #break

            
    return fstats_tot

def balance_data(df, target, **kwargs):
    """
    Balances the dataset so there are equal number of rows for each class
    Parameters:
    ----------
    df: pandas.DataFrame
        dataframe to be balanced
    target: string
        name of dataframe column that represents that class the row is from

    Returns:
    --------
    bal_df: pandas.DataFrame
        dataframe with equal number of rows per unique class
    """
    if 'random_state' not in kwargs:
        random_state = 1
    else:
        random_state = kwargs['random_state']
    df_target = []
    bal_df = []
    for name in df[target].unique():
        df_target.append((name, df[df[target] == name]))
    print(f"Ratio before data balance ({':'.join([str(i[0]) for i in df_target])}) = {':'.join([str(len(i[1])) for i in df_target])}")
    for i in range(len(df_target)):
        ratio = min([len(i[1]) for i in df_target])/len(df_target[i][1])
        bal_df.append(df_target[i][1].sample(frac=ratio, random_state=random_state))
    print(f"Ratio after balance ({':'.join([str(i[0]) for i in df_target])}) = {':'.join([str(len(i)) for i in bal_df])}")
    return pd.concat(bal_df)


def bin_data(bal_ecm, resolution=128):
    """
    Takes in a dataframe that has a binx and a biny column, and uses
    those columns to generate a bin column based on the resolution
    This is necessary for eventual cross validation to prevent data leakage

    Parameters
    ----------
    bal_ecm: pandas.DataFrame
        dataframe to be processed. Dataframe may need to have balanced classes - use balance_data function
    resolution: int
        integer representing the size of the bins. Resolution must be a factor of 2048 and > 128
        default is 128

    Returns
    -------
    bal_ecm: pandas.DataFrame
        dataframe with new column indicating which bin a give row is in
    """
    assert not 2048%resolution and resolution >= 128, "resolution needs to be a factor of 2048 and > 128"
    bins = list(range(0, 2048+1, resolution))
    bin_labels = [int(i/resolution) for i in bins][:-1]
    bal_ecm['binx'] = pd.cut(bal_ecm['X'], bins, labels=bin_labels, include_lowest=True)
    bal_ecm.loc[bal_ecm['X'] < 0] = 0
    bal_ecm['biny'] = pd.cut(bal_ecm.Y, bins, labels=bin_labels, include_lowest=True)
    bal_ecm['bins'] = (len(bins)-1)*bal_ecm['binx'].astype(np.int32) + bal_ecm['biny'].astype(np.int32)
    bal_ecm = bal_ecm[np.isfinite(bal_ecm['bins'])]
    bal_ecm['bins'] = bal_ecm['bins'].astype(int)
    return bal_ecm

# This might be redundant, fine for now
subscription_id = '5b7e9376-1907-45b5-b8cf-6fde28c54e67'
resource_group  = 'mpt_projects'
workspace_name  = 'ecm_project'
workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='age_data')
dataset.download(target_path='.', overwrite=True)

datasetpath = getcwd()
print('Current Notebook Dir: ' + workbookDir)

# for file in dataset.to_path():
#     print(file)
#     df = pd.read_csv(workbookDir + file)
#     print(df.shape)
filelist = [f for f in listdir(dataset.to_path()) if isfile(join(dataset.to_path(), f)) and 'feat' in f]

sampled_filelist = []
class_lens = [0, 15, 30, 45]
for i in range(len(class_lens)-1):
    rand_integers = random.sample(set(np.arange(class_lens[i], class_lens[i+1])), 15)
    for rand_int in rand_integers:
        sampled_filelist.append(filelist[rand_int])

fstats_tot = generate_fullstats(dataset_path, sampled_filelist, ['P14', 'NT', 'P70'], 'age')
print(fstats_tot).head()

# features = [
#     'alpha', # Fitted anomalous diffusion alpha exponenet
#     'D_fit', # Fitted anomalous diffusion coefficient
#     'kurtosis', # Kurtosis of track
#     'asymmetry1', # Asymmetry of trajecory (0 for circular symmetric, 1 for linear)
#     'asymmetry2', # Ratio of the smaller to larger principal radius of gyration
#     'asymmetry3', # An asymmetric feature that accnts for non-cylindrically symmetric pt distributions
#     'AR', # Aspect ratio of long and short side of trajectory's minimum bounding rectangle
#     'elongation', # Est. of amount of extension of trajectory from centroid
#     'boundedness', # How much a particle with Deff is restricted by a circular confinement of radius r
#     'fractal_dim', # Measure of how complicated a self similar figure is
#     'trappedness', # Probability that a particle with Deff is trapped in a region
#     'efficiency', # Ratio of squared net displacement to the sum of squared step lengths
#     'straightness', # Ratio of net displacement to the sum of squared step lengths
#     'MSD_ratio', # MSD ratio of the track
# #     'frames', # Number of frames the track spans
#     'Deff1', # Effective diffusion coefficient at 0.33 s
#     'Deff2', # Effective diffusion coefficient at 3.3 s
#     #'angle_mean', # Mean turning angle which is counterclockwise angle from one frame point to another
#     #'angle_mag_mean', # Magnitude of the turning angle mean
#     #'angle_var', # Variance of the turning angle
#     #'dist_tot', # Total distance of the trajectory
#     #'dist_net', # Net distance from first point to last point
#     #'progression', # Ratio of the net distance traveled and the total distance
#     'Mean alpha', 
#     'Mean D_fit', 
#     'Mean kurtosis', 
#     'Mean asymmetry1', 
#     'Mean asymmetry2',
#     'Mean asymmetry3', 
#     'Mean AR',
#     'Mean elongation', 
#     'Mean boundedness',
#     'Mean fractal_dim', 
#     'Mean trappedness', 
#     'Mean efficiency',
#     'Mean straightness', 
#     'Mean MSD_ratio', 
#     'Mean Deff1', 
#     'Mean Deff2',
#     ]

# target = 'age'

# ecm = fstats_tot[features + [target, 'Track_ID', 'X', 'Y']]
# ecm = ecm[~ecm[list(set(features) - set(['Deff2', 'Mean Deff2']))].isin([np.nan, np.inf, -np.inf]).any(1)]       # Removing nan and inf data points
# bal_ecm = balance_data(ecm, target)
# sampled_df = bin_data(bal_ecm)
# label_df = sampled_df['age']
# features_df = sampled_df.drop(['age', 'X', 'Y', 'binx', 'biny', 'bins', 'Track_ID'], axis=1)
# features = features_df.columns

# seed = 1234
# np.random.seed(seed)
# train_split = 0.5
# test_split = 0.5

# le = preprocessing.LabelEncoder()
# sampled_df['encoded_target'] = le.fit_transform(sampled_df[target])

# training_bins = np.random.choice(sampled_df['bins'].unique(), int(len(sampled_df['bins'].unique())*train_split), replace=False)

# X_train = sampled_df[sampled_df['bins'].isin(training_bins)]
# X_test_val = sampled_df[~sampled_df['bins'].isin(training_bins)]
# X_val, X_test = train_test_split(X_test_val, test_size=test_split, random_state=seed)

# y_train = X_train['encoded_target']
# y_test = X_test['encoded_target']
# y_val = X_val['encoded_target']

# dtrain = xgb.DMatrix(X_train[features], label=y_train)
# dtest = xgb.DMatrix(X_test[features], label=y_test)
# dval = xgb.DMatrix(X_val[features], label=y_val)

# param = {'max_depth': 3,
#          'eta': 0.005,
#          'min_child_weight': 0,
#          'verbosity': 0,
#          'objective': 'multi:softprob',
#          'num_class': 3,
#          'silent': 'True',
#          'gamma': 5,
#          'subsample': 0.15,
#          'colsample_bytree': 0.8,
#          'eval_metric': "mlogloss",
#          # GPU integration will cut time in ~half:
#          'gpu_id' : 0,
#          'tree_method': 'gpu_hist',
#          'predictor': 'gpu_predictor'
#          }

# (best_model, best_param, best_eval, best_boost_rounds) = predxgboost.xgb_paramsearch(X_train, y_train, features, init_params=param, nfold=5, num_boost_round=2000, early_stopping_rounds=3, use_gpu='True')

# booster, acc, true_label, preds = train(best_param, dtrain, dtest, dval, evals=[(dtrain, 'train'), (dval, 'eval')], num_round=1000)

# Save a sample of the data in the outputs folder (which gets uploaded automatically)
##os.makedirs('outputs', exist_ok=True)
#data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
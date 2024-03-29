import numpy as np
import pandas as pd
import xgboost as xgb
from diff_predictor.predxgboost import bin_fold, mknfold, cv, aggcv, xgb_paramsearch
from diff_predictor.data_process import bin_data, split_data
from hypothesis import example, given, strategies as st

# Creating reproduceable dataframe for testing
np.random.seed(1234)
param = {}
categories = ['alpha', 'D_fit', 'kurtosis', 'asymmetry1', 'asymmetry2',
              'asymmetry3', 'AR', 'elongation', 'boundedness', 'fractal_dim',
              'trappedness', 'efficiency', 'straightness', 'MSD_ratio',
              'frames', 'Deff1', 'Deff2', 'angle_mean', 'angle_mag_mean',
              'angle_var', 'dist_tot', 'dist_net', 'progression',
              'Mean alpha', 'Mean D_fit', 'Mean kurtosis', 'Mean asymmetry1',
              'Mean asymmetry2', 'Mean asymmetry3', 'Mean AR',
              'Mean elongation', 'Mean boundedness', 'Mean fractal_dim',
              'Mean trappedness', 'Mean efficiency', 'Mean straightness',
              'Mean MSD_ratio', 'Mean Deff1', 'Mean Deff2']
df = pd.DataFrame(np.random.randn(1000, len(categories)), columns=categories)
df['target'] = np.random.choice([0, 1, 2, 3], 1000, p=[0.1, 0.4, 0.2, 0.3])
df['X'] = np.random.uniform(0.0, 2048.0, 1000)
df['Y'] = np.random.uniform(0.0, 2048.0, 1000)
param = {'max_depth': 3,
         'eta': 0.005,
         'min_child_weight': 0,
         'verbosity': 0,
         'objective': 'multi:softprob',
         'num_class': 4,
         'silent': 'True',
         'gamma': 5,
         'subsample': 0.15,
         'colsample_bytree': 0.8,
         'eval_metric': "mlogloss"}
df_bins = bin_data(df, resolution=128) ##
df_split, le = split_data(df_bins, 'target', 0.75) ##
X_train, y_train, X_test, y_test = df_split
nfold = 5

def test_bin_fold():
    df_fold, wt = bin_fold(df_bins, nfold)
    assert sum(wt) == 1, "Weights of bins do not add up to 1.0"
    assert len(wt) == nfold, "Outputed the incorrect number of folds"
    assert len(df_fold) == nfold, "Outputed the incorrect number of folds"
    assert sum(len(df_fold[i]) for i in range(5)) == len(df), "Did not bin all data"
    
    
def test_mknfold():
    df_bins = bin_data(df, resolution=128)
    ret, wt = mknfold(X_train, y_train, nfold, param, features=categories)
    assert sum(wt) == 1, "Weights of bins to not add up to 1.0"
    assert len(wt) == nfold, "Outputed the incorrect number of folds"
    assert len(ret) == nfold, "Outputed the incorrect number of folds"
    assert all(isinstance(i, xgb.training.CVPack) for i in ret), "return value not xgb.training.CVPack datatype"
    
    
def test_cv():
    cv_model = cv(param, X_train, y_train, features=categories, num_boost_round=50, nfold=nfold)
    assert isinstance(cv_model, pd.core.frame.DataFrame), "Incorrect datatype"
    assert all(cv_model == ['train-mlogloss-mean', 'train-mlogloss-std', 'test-mlogloss-mean',
                'test-mlogloss-std']), "Incorrect metric columns"
    
    
def test_aggcv():
    feval=None
    i = 1
    ret, wt = mknfold(X_train, y_train, nfold, param, features=categories)
    res = aggcv([f.eval(i, feval) for f in ret], wt)
    assert len(res) == 2, "aggregate results not found for both training and testing"
    assert np.array(res).shape == (2, 3), "result needs label, mean, and std for each train and testing metric"

    
def test_xgb_paramsearch():
    best_model, best_param, best_eval, best_boost_rounds = xgb_paramsearch(X_train = X_train,
                                                                           y_train = y_train,
                                                                           features = categories,
                                                                           init_params = param,
                                                                           nfold=nfold,
                                                                           num_boost_round=5,
                                                                           early_stopping_rounds=3,
                                                                           metrics=['mlogloss'],
                                                                           seed=1234,
                                                                           early_break=2)
#     assert len(xgb_ps) == 4, "calculated results not right length"
    return best_model, best_param, best_eval, best_boost_rounds
    
    
    
def test_train():
    pass
    
    
def test_save():
    pass
    
    
def test_load():
    pass
    
    
def test_get_dmatrix():
    pass
    
    
def test_get_params():
    pass

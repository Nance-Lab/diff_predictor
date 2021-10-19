from attr import s
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from os import listdir, getcwd, chdir
from os.path import isfile, join
from diff_predictor.eval import perf_meas, corrmat
from hypothesis import given, strategies as st, example
from hypothesis.extra.pandas import columns, column, data_frames, range_indexes

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


data_cols = columns(names_or_number=categories, dtype=float, elements=st.floats())
position_cols = columns(names_or_number=['X', 'Y'], dtype=float, elements=st.floats(min_value=0.0, max_value=2048.0))
target_col = column(name='target', dtype=int, elements=st.integers(min_value=0, max_value=20)) #up to twenty unique targets

df = data_frames(columns=data_cols + position_cols + [target_col], index=range_indexes(min_size=1))


strat1 = st.integers(0,1)
strat2 = st.integers(0,10)
#size = st.integers(min_value=1)

#y_pred_and_actual = st.tuples(st.lists(strat1, min_size=size, max_size=size), st.lists(strat1, min_size=size, max_size=size))

@given(s=st.lists(strat1), q=st.integers(0,1))
def test_perf_meas(s, q):
    y_actual = list(s)
    y_pred=random.sample(s, len(s))
    (TP, FP, TN, FN) = perf_meas(y_actual=y_actual, y_pred=y_pred, cls=q)
    assert TP+FP+TN+FN == len(y_pred)

@given(s=df)
def test_corrmat(s):
    corr_mat = corrmat(s, show_plot=False)
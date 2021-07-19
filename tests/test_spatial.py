import numpy as np
import pandas as pd
import xgboost as xgb
from diff_predictor.predxgboost import bin_fold, mknfold, cv, aggcv, xgb_paramsearch
from diff_predictor.data_process import bin_data
from diff_predictor.spatial import split_data, balance_data, get_lengths
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import columns


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

data_strat = st.floats()

testing_cols = columns(names_or_number=categories, dtype=float, elements=data_strat)


def test_balance_data():
   pass

print('so far so good')
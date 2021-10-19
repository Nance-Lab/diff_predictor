import numpy as np
import pandas as pd
import xgboost as xgb
from os import listdir, getcwd, chdir
from os.path import isfile, join
from diff_predictor.data_process import split_data, balance_data, generate_fullstats, bin_data
from hypothesis import given, strategies as st
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

@given(s=df)
def test_balance_data(s):
    bal_df = balance_data(df=s, target='target')
    assert (len(bal_df['target'].unique()) == len(s['target'].unique())) # check there are the same number of unique targets
    class_list = bal_df['target'].unique()
    bal_length = len(bal_df[bal_df['target'] == class_list[0]]) # get the count of a specific class
    for val in class_list:
        assert len(bal_df[bal_df['target'] == val] == bal_length)

@given(s=df)
def test_bin_data(s):
    static_df = s.copy() #needed because bin_data edits the dataframe, doesnt return a new one
    bin_df = bin_data(s)
    assert len(static_df.columns) + 3 == len(bin_df.columns) # we are expecting to add three columns
    assert bin_df['bins'].isnull().values.any() == False # all trajectories should have a bin
    #assert bin_df['bins'].max() > 0
    #assert len(bin_df['bins'].unique()) > 1

@given(s=df)
def test_split_data(s):
    bin_df = bin_data(s)
    result, le = split_data(df=bin_df, target='target', train_split=0.5)

dataset_path = '../diff_predictor/tests/testing_data/test_generate_fullstats_data/'
filelist = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and 'feat' in f]
def test_generate_fullstats():
    full_stats = generate_fullstats(dataset_path=dataset_path, filelist=filelist, targets=['NT', 'P14', 'P21', 'P28'])
    assert len(full_stats) > 0
    assert len(full_stats['Target'].unique()) == 4
    assert set(['NT', 'P14', 'P21', 'P28']).issubset(full_stats['Target'].unique())
    assert full_stats['Video Number'].max() == len(filelist)-1 # -1 needed due to 0 indexing

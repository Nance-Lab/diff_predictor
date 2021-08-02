#Script for loading, processing, splitting, and saving an MPT dataset
#Useful for generating reproducible test, train, val csv file
#or generating files to upload to other platforms (ie Google Colab)

from diff_predictor.data_process import generate_fullstats, balance_data, bin_data, split_data
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

dataset_path = '/Users/nelsschimek/Documents/Nance Lab/diff_predictor/data/raw_data_age/'
filelist = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and 'feat' in f]

feature_list = [
    'alpha', # Fitted anomalous diffusion alpha exponenet
    'D_fit', # Fitted anomalous diffusion coefficient
    'kurtosis', # Kurtosis of track
    'asymmetry1', # Asymmetry of trajecory (0 for circular symmetric, 1 for linear)
    'asymmetry2', # Ratio of the smaller to larger principal radius of gyration
    'asymmetry3', # An asymmetric feature that accnts for non-cylindrically symmetric pt distributions
    'AR', # Aspect ratio of long and short side of trajectory's minimum bounding rectangle
    'elongation', # Est. of amount of extension of trajectory from centroid
    'boundedness', # How much a particle with Deff is restricted by a circular confinement of radius r
    'fractal_dim', # Measure of how complicated a self similar figure is
    'trappedness', # Probability that a particle with Deff is trapped in a region
    'efficiency', # Ratio of squared net displacement to the sum of squared step lengths
    'straightness', # Ratio of net displacement to the sum of squared step lengths
    'MSD_ratio', # MSD ratio of the track
#     'frames', # Number of frames the track spans
    'Deff1', # Effective diffusion coefficient at 0.33 s
    'Deff2', # Effective diffusion coefficient at 3.3 s
    #'angle_mean', # Mean turning angle which is counterclockwise angle from one frame point to another
    #'angle_mag_mean', # Magnitude of the turning angle mean
    #'angle_var', # Variance of the turning angle
    #'dist_tot', # Total distance of the trajectory
    #'dist_net', # Net distance from first point to last point
    #'progression', # Ratio of the net distance traveled and the total distance
    'Mean alpha', 
    'Mean D_fit', 
    'Mean kurtosis', 
    'Mean asymmetry1', 
    'Mean asymmetry2',
    'Mean asymmetry3', 
    'Mean AR',
    'Mean elongation', 
    'Mean boundedness',
    'Mean fractal_dim', 
    'Mean trappedness', 
    'Mean efficiency',
    'Mean straightness', 
    'Mean MSD_ratio', 
    'Mean Deff1', 
    'Mean Deff2',
    ]

target = 'age'

fstats_tot = generate_fullstats(dataset_path, filelist, ['P14', 'P70'], target)
#dropped_fstats = fstats_tot.drop_duplicates(['Mean Deff1', 'Mean D_fit'])
ecm = fstats_tot[feature_list + [target, 'Track_ID', 'X', 'Y']]
ecm = ecm[~ecm[list(set(feature_list) - set(['Deff2', 'Mean Deff2']))].isin([np.nan, np.inf, -np.inf]).any(1)]       # Removing nan and inf data points
bal_ecm = balance_data(ecm, target)
sampled_df = bin_data(bal_ecm)
result, le = split_data(sampled_df, target, 0.7, 0.5)
print(f'length of training data: {len(result[0])}')
print(result[0].columns)
print(f'length of testing data: {len(result[2])}')
print(f'length of validation data: {len(result[4])}')

train = result[0][feature_list + ['encoded_target']]
train = train.rename(columns={'encoded_target': 'Y'})

test = result[2][feature_list + ['encoded_target']]
test = test.rename(columns={'encoded_target': 'Y'})

valid = result[4][feature_list + ['encoded_target']]
valid = valid.rename(columns={'encoded_target': 'Y'})

print(train.columns)

train.to_csv('/Users/nelsschimek/Documents/Nance Lab/diff_predictor/data/dvrl_data/age_data_binary/train.csv')
test.to_csv('/Users/nelsschimek/Documents/Nance Lab/diff_predictor/data/dvrl_data/age_data_binary/test.csv')
valid.to_csv('/Users/nelsschimek/Documents/Nance Lab/diff_predictor/data/dvrl_data/age_data_binary/valid.csv')
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

import data_process
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from os import listdir, getcwd, chdir
from os.path import isfile, join

from sklearn.preprocessing import scale, StandardScaler
from numpy.random import permutation

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, precision_score, f1_score

import operator
import shap

from sklearn_model_wrapper import paramsearch, train, test

data_dir = getcwd() + '/testing_data/'

filelist = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and 'feat' in f]
assert(len(filelist) == 15)

# generate single csv from filelist
fstats_tot_age = data_process.generate_fullstats(data_dir, filelist, ['P14', 'P35', 'P70'], 'age')

feature_list = [
    'alpha',        # Fitted anomalous diffusion alpha exponenet
    'D_fit',        # Fitted anomalous diffusion coefficient
    'kurtosis',     # Kurtosis of track
    'asymmetry1',   # Asymmetry of trajecory (0 for circular symmetric, 1 for linear)
    'asymmetry2',   # Ratio of the smaller to larger principal radius of gyration
    'asymmetry3',   # An asymmetric feature that accnts for non-cylindrically symmetric pt distributions
    'AR',           # Aspect ratio of long and short side of trajectory's minimum bounding rectangle
    'elongation',   # Est. of amount of extension of trajectory from centroid
    'boundedness',  # How much a particle with Deff is restricted by a circular confinement of radius r
    'fractal_dim',  # Measure of how complicated a self similar figure is
    'trappedness',  # Probability that a particle with Deff is trapped in a region
    'efficiency',   # Ratio of squared net displacement to the sum of squared step lengths
    'straightness', # Ratio of net displacement to the sum of squared step lengths
    'MSD_ratio',    # MSD ratio of the track
    'Deff1',        # Effective diffusion coefficient at 0.33 s
    'Deff2',        # Effective diffusion coefficient at 3.3 s
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
    'Mean Deff2'
]

target = 'age'
NUM_CLASSES = 3

ecm = fstats_tot_age[feature_list + [target, 'Track_ID', 'X', 'Y']] # ecm=extra cellular matrix, include features, columns X Y
# get indexes, important for binning, dont need std dev
ecm = ecm[~ecm[list(set(feature_list) - set(['Deff2', 'Mean Deff2']))].isin([np.nan, np.inf, -np.inf]).any(axis=1)]  # Remove nan and inf data points
print('ecm shape:', ecm.shape)

bal_ecm = data_process.balance_data(ecm, target, random_state=1)
bal_ecm = data_process.bin_data(bal_ecm, resolution=128)
label_df = bal_ecm[target]
features_df = bal_ecm.drop([target, 'Track_ID', 'X', 'Y', 'binx', 'biny', 'bins'], axis=1)
features = features_df.columns

# Regular split

seed = 1234
np.random.seed(seed)
train_split = 0.8
test_split = 0.5

le = preprocessing.LabelEncoder()
bal_ecm['encoded_target'] = le.fit_transform(bal_ecm[target])

training_bins = np.random.choice(bal_ecm.bins.unique(), int(len(bal_ecm.bins.unique())*train_split), replace=False)

training_bins = np.random.choice(bal_ecm.bins.unique(), int(
    len(bal_ecm.bins.unique())*train_split), replace=False)

X_train = bal_ecm[bal_ecm.bins.isin(training_bins)]
X_test_val = bal_ecm[~bal_ecm.bins.isin(training_bins)]
X_val, X_test = train_test_split(X_test_val, test_size=test_split, random_state=seed)

y_train = X_train['encoded_target']
y_test = X_test['encoded_target']
y_val = X_val['encoded_target']

dtrain = X_train[features]
# X_train has features we don't care about
dtest = X_test[features]
dval = X_val[features]


# is spatial module actually needed? it's not defined anywhere
# get_lengths-do sizes of outputs make sense (splitting)
# TODO print len of each; make test

print(f'Tot before split: {len(bal_ecm)}')
print(f'Training: {len(X_train)} ({len(X_train) / len(bal_ecm):.3f}%)')
print(f'Testing: {len(X_test)} ({len(X_test) / len(bal_ecm):.3f}%)')
print(f'Evaluation: {len(X_val)} ({len(X_val) / len(bal_ecm):.3f}%)')

# param = {'max_depth': 3,
#          'eta': 0.005,
#          'min_child_weight': 0,
#          'verbosity': 0,
#          'objective': 'multi:softprob',
#          'num_class': 3,
#          'silent': 'True', how much info it tells you ab model being trained; output
#          'gamma': 5,
#          'subsample': 0.15, less important
#          'colsample_bytree': 0.8, less important
#          'eval_metric': "mlogloss"}

param = {
    'max_depth' : (1, 10, False),
    'eta': (0.005, 0.2, True),
    'min_child_weight': (0, 1, False),
    'gamma': (1, 6, False)}

rf_model = RandomForestClassifier()
# best_param = paramsearch(rf_model, param, dtrain, y_train, dval, y_val)
# print('best hyperparameter values:', best_param)
temp_best_param = {'max_depth': 7, 'eta': 0.012917148675308392, 'min_child_weight': 1, 'gamma': 6}

# # train model, make predictions
trained_model, train_model = train(rf_model, temp_best_param, NUM_CLASSES, dtrain, y_train)
# test_acc, test_pred = test(model, dtest, y_test)

# TODO try diff type of classifier: SVM
svc_model = SVC()
lin_svc_model = LinearSVC()
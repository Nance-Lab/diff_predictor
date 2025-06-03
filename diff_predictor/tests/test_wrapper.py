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

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.utils import all_estimators

import operator
import shap

from sklearn_model_wrapper import paramsearch, train, test

import pdb

data_dir = getcwd() + '/testing_data/'

filelist = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and 'feat' in f]
assert(len(filelist) == 15)

# Generate single csv from filelist
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
]

target = 'age'
NUM_CLASSES = 3

ecm = fstats_tot_age[feature_list + [target, 'Track_ID', 'X', 'Y']] # ecm=extra cellular matrix, include features, columns X Y
# Get indexes, important for binning, dont need std dev

ecm = ecm[~ecm[feature_list].isin([np.nan, np.inf, -np.inf]).any(axis=1)]  # Remove nan and inf data points
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
dtest = X_test[features]
dval = X_val[features]

print(f'Tot before split: {len(bal_ecm)}')
print(f'Training: {len(X_train)} ({len(X_train) / len(bal_ecm):.3f}%)')
print(f'Testing: {len(X_test)} ({len(X_test) / len(bal_ecm):.3f}%)')
print(f'Evaluation: {len(X_val)} ({len(X_val) / len(bal_ecm):.3f}%)')

param = {
    'max_depth' : (1, 10, False),
    'eta': (0.005, 0.2, True),
    'min_child_weight': (0, 1, False),
    'gamma': (1, 6, False)
}

ensembles = [
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
]

svms = [
    NuSVC,
    SVC
]

trees = [
    DecisionTreeClassifier,
    ExtraTreeClassifier
]

model_types = ensembles + svms + trees

actual_test = np.array(y_test.tolist())

accuracies = []

for model_type in model_types:
    print(f"Model type: {model_type}")
    trained_model = train(model_type, None, NUM_CLASSES, dtrain, y_train)
    test_acc, test_pred = test(trained_model, dtest, y_test)
    accuracies.append(test_acc)
    print(f"Accuracy:{test_acc * 100: .2f}%")
    print(f"Predictions: {test_pred}")
    print(f"Actual     : {actual_test}")

model_names = [
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'GradientBoostingClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
    'NuSVC',
    'SVC',
    'DecisionTreeClassifier',
    'ExtraTreeClassifier'
]

fig, ax = plt.subplots()
y_pos = np.arange(len(model_names))
accuracies = np.array(accuracies)

ax.barh(y_pos, accuracies, align='center')
ax.set_yticks(y_pos, labels=model_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy')
ax.set_title('Sklearn Models vs. Test Accuracy')

plt.show()
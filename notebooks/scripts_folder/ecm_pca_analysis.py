from azureml.core import Run
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from os import listdir, getcwd, chdir
from os.path import isfile, join
import os
from diff_predictor import core, data_process
from diff_classifier import pca
from matplotlib import colors as plt_colors



from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Get the experiment run context
run = Run.get_context()

# load the  dataset
dataset_path = 'Users/nlsschim/diff_predictor/data/0.8s_deff2_features_ecm'
filelist = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and 'feat' in f]
fstats_tot = data_process.generate_fullstats(dataset_path, filelist, ['ChABC','NT'], 'Treatment')

columns = [
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
   'frames', # Number of frames the track spans
    'Deff1', # Effective diffusion coefficient at 0.33 s
    'Deff2', # Effective diffusion coefficient at 3.3 s
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
    'Treatment',
    'X',
    'Y'
]

target = 'Treatment'

ecm = fstats_tot[columns]#.drop(['Deff2', 'Mean Deff2', 'Std Deff2'], axis=1) # Removing since 97% is null
#ecm['Treatment'] = fstats_tot['Treatment']
ecm = ecm[~ecm.isin([np.nan, np.inf, -np.inf]).any(1)] # removes rows with nan or inf points

bal_ecm = ecm

label_df = bal_ecm['Treatment']
features_df = bal_ecm.drop(['Treatment', 'X', 'Y', 'frames'], axis=1).astype(float)
#df = df.astype(float)

# Data needs to be scaled for correlation, t-SNE, and PCA analysis
ss = StandardScaler()
scaled_data = pd.DataFrame(ss.fit_transform(features_df.values), columns=features_df.columns)

scaled_data = scale(scaled_data, axis=1)
scaled_df = pd.DataFrame(scaled_data, columns = features_df.columns)
scaled_df['Treatment'] = label_df.values
scaled_df.mean()

pcadataset = pca.pca_analysis(ecm, dropcols=['Treatment', 'X', 'Y', 'frames'], n_components=10, scale=True)

      
# Save a sample of the data in the outputs folder (which gets uploaded automatically)
##os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
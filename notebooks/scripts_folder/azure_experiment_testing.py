from azureml.core import Run
import boto3
import pandas as pd
import numpy as np
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
from xgboost.training import CVPack
from xgboost import callback
from xgboost.core import CallbackEnv
from xgboost.core import EarlyStopException
from xgboost.core import STRING_TYPES

# Get the experiment run context
run = Run.get_context()



# Save a sample of the data in the outputs folder (which gets uploaded automatically)
##os.makedirs('outputs', exist_ok=True)
#data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
import sys
import operator
import pandas as pd
import numpy as np
import opendataval
import torch
import optuna
import sklearn

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

from opendataval.model import ClassifierSkLearnWrapper

import pdb

'''
Wrapper module that allows a user to choose any model architecture
listed on scikit-learn, and train it on MPT data. Aims to ensure that,
regardless of what model is used, hyperparameter tuning and train test
validation splitting appropriately have no leakage of trajectories that
are in the same microenvironment.
'''

def paramsearch(models, param, X_train, y_train, X_val, y_val):
    '''
    Performs hyperparameter tuning and returns the best hyperparameters found for the given model.

    Parameters
    ----------
    model : list
        List with model types from scikit-learn to find hyperparameters for
    param : dict
        Dictionary of hyperparameters and their ranges to test out.
        Should be formatted like: {hyperparam: (low, high, log_val)}
        e.g. param = {'learning_rate_init': (1e-5, 1e-3, True)}
    X_train : pandas.DataFrame
        Training data for fitting.
    y_train : pandas.DataFrame
        Training data for fitting.
    X_val: pandas.DataFrame
        Validation data for hyperparameter tuning.
    y_val: pandas.DataFrame
        Validation data for hyperparameter tuning.

    Returns
    -------
    model: scikit-learn model
        The trained model with the best hyperparameter values.
    best_params :
        The best values found for the hyperparameters given.
    '''
    best_params = []

    for model_type in models:

        def objective(trial):
            '''
            trial : optuna.trial
                Used for suggesting values for hyperparameters.
            '''
            for hyperparam, val in param.items():
                low, high, log_TF = val[0], val[1], val[2]
                if isinstance(low, int):
                    suggested_val = trial.suggest_int(hyperparam, low, high, log=log_TF)
                    setattr(model, hyperparam, suggested_val)
                elif isinstance(low, float):
                    suggested_val = trial.suggest_float(hyperparam, low, high, log=log_TF)
                    setattr(model, hyperparam, suggested_val)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)

        model = model_type()
        study = optuna.create_study(direction='maximize') # ok to maximize accuracy
        study.optimize(objective, n_trials=100)
        best_params.append(study.best_params)

    return best_params

def train(model_type, param, num_class: int, X_train : pd.DataFrame, y_train : pd.DataFrame,
          dval=False, evals=None, num_round=2000, verbose=True):
    '''
    Trains a model listed on scikit-learn on MPT data and returns the trained model.

    Parameters
    ----------
    model_type : model type from scikit-learn
        Model to be trained and returned.
    param : dict
        Dictionary of parameters used for training.
    num_class : int
        Number of classes to use
    X_train : pandas.DataFrame
        Training data for fitting.
    y_train : pandas.DataFrame
        Training data for fitting.

    Optional Parameters
    -------------------
    evals : list : None
        Evaluation configuration. Will report results in this form. If dval is
        used, will automatically update to [(dtrain, `train`), (dval, `eval`)].
        Will use the last evaluation value in the list to test for loss
        convergence
    (xgboost) dval : xgboost.DMatrix : None
        Evaluation data for fitting.
    (xgboost) num_rounds : int : 2000
        Number of boosting rounds to go through when training. A higher number
        makes a more complex ensemble model.
    (xgboost) num_boost_round : int : None
        Number of boosting iterations.

    Returns
    -------
    model : opendataval.ClassifierSkLearnWrapper
        Resulting trained model.
    '''
    # TODO: how to set model's hyperparameters? using setattr() or through kwargs?
    # breakpoint()
    module = getattr(model_type, '__module__', '')
    if module.startswith('sklearn.svm'):
        wrapped_model = ClassifierSkLearnWrapper(model_type, num_class, probability=True)
    else:
        wrapped_model = ClassifierSkLearnWrapper(model_type, num_class)

    # Convert given data to Tensors
    X_train_tensor = torch.tensor(X_train.values)
    y_train_tensor = torch.tensor(y_train.values)

    # Ensure y_train_tensor is in the correct format
    if y_train_tensor.ndimension() == 1:
        y_train_one_hot = torch.nn.functional.one_hot(y_train_tensor, num_classes=num_class).float()
    else:
        y_train_one_hot = y_train_tensor.float()

    # Train by calling wrapper's fit()
    wrapped_model.fit(X_train_tensor, y_train_one_hot)

    return wrapped_model

def test(model, X_test, y_test):
    '''
    Tests a model listed on scikit-learn on MPT data and returns the results.

    Parameters
    ----------
    model : opendataval.ClassifierSkLearnWrapper
        Model on which to test predictions.
    X_test : pandas.DataFrame
        Labels to predict.
    y_test : pandas.DataFrame
        True labels.

    Returns
    -------
    acc : float
        Accuracy of model.
    preds: Tensor
        Predictions from model.
    '''
    X_test_tensor = torch.tensor(X_test.values)

    # Call wrapper's predict()
    y_pred_tensor = model.predict(X_test_tensor)

    # Convert predictions to numpy
    y_pred_np = y_pred_tensor.numpy()

    # Get the max value of the row, convert to 0, 1, or 2 depending on the index of max value
    y_classes = np.argmax(y_pred_np, axis=1)

    acc = accuracy_score(y_test, y_classes)  # y_test is true label

    return acc, y_classes
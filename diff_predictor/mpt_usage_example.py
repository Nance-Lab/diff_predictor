import sklearn_model_for_mpt as mpt
import sys
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from xgboost import Booster, XGBClassifier

if __name__ == '__main__':

  X_train, y_train, features, init_params = None, None, None, None
  nfold, early_stopping_rounds = 5, 3
  model_type = rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

  best_model, best_param, best_eval = mpt.paramsearch(X_train, y_train, \
                                                      features, init_params)  # return metrics

  dtrain, dtest = None, None
  # evals = [(dtrain, 'train'), (dval, 'eval')]
  # num_round = 804

  rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
  booster = xgb.Booster()
  sklearn_xgb = xgb.XGBClassifier()

  # train model/make predictions
  trained_rf_classifier, accuracy = mpt.train(rf_classifier, best_param, dtrain, dtest)
  trained_booster, accuracy = mpt.train(booster, best_param, dtrain, dtest)
  trained_booster, accuracy = mpt.train(sklearn_xgb, best_param, dtrain, dtest)

  y_test, preds = None, None
  metrics.confusion_matrix(y_test, preds) # get confusion matrix

  # do shaft analysis - TODO: where to look to understand how to do this?

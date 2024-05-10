import sklearn_model_for_mpt as mpt
import sys
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from xgboost import Booster, XGBClassifier

if __name__ == '__main__':

  # TODO: give these values
  X_train, y_train = None, None
  X_val, y_val = None, None
  X_test, y_test = None

  hparam = {'max_depth' : (1e-10, 1e10, True)}
  model = rf_classifier = RandomForestClassifier()
  setattr(rf_classifier, 'n_estimators', 10)

  # xgboost-specific:
  # dtrain, dtest = None, None
  # evals = [(dtrain, 'train'), (dval, 'eval')]
  # num_round = 804
  # booster = xgb.Booster()
  # sklearn_xgb = xgb.XGBClassifier()
  # nfold, early_stopping_rounds = 5, 3

  # hyperparameter tuning
  best_model, best_param = mpt.paramsearch(model, X_train, y_train, X_val, y_val, hparam)

  # train model/make predictions
  trained_model, train_accuracy = mpt.train(best_model, X_train, y_train)
  tested_model, test_accuracy = mpt.test(best_model, X_test, y_test)


  # metrics.confusion_matrix(y_test, preds) # get confusion matrix

  # TODO later: shap analysis
  # TODO later: write test functions

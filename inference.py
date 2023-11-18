import pandas as pd
import xgboost as xgb
from data_preparation import data_prep
import joblib

PATH_DATA = "data"
MODEL_PATH = "model_1.pkl"

params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,

    'gamma': 0,
    'lambda': 0,
    'alpha': 0,
    'min_child_weight': 0,

    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'njobs': -1,
    'tree_method': 'approx'
}



def predict(loaded_model, train, test):
    loaded_model = joblib.load(MODEL_PATH)
    y_pred = loaded_model.predict(xgb.DMatrix(test.values, feature_names=list(train.columns)))
    submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])

    return submission


data_train, data_test, target = data_prep(PATH_DATA)

loaded_model = joblib.load(MODEL_PATH)

submission = predict(loaded_model, data_train, data_test)

submission.to_csv('result.csv')


# def cv_score(params, train, y_true):
#     cv_res=xgb.cv(params, xgb.DMatrix(train, y_true),
#                   early_stopping_rounds=10, maximize=True,
#                   num_boost_round=10000, nfold=5, stratified=True)
#     index_argmax = cv_res['test-auc-mean'].argmax()
#     print('Cross-validation, ROC AUC: {:.3f}+-{:.3f}, Trees: {}'.format(cv_res.loc[index_argmax]['test-auc-mean'],
#                                                                         cv_res.loc[index_argmax]['test-auc-std'],
#                                                                         index_argmax))
#
#
# cv_score(params, data_train, target)
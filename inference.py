import pandas as pd
import xgboost as xgb
from data_preparation import data_prep
import joblib
from catboost import CatBoostClassifier

PATH_DATA = "data"
MODEL_PATH = "model_2.pkl"

data_train, data_test, target = data_prep(PATH_DATA)

loaded_model = joblib.load(MODEL_PATH)

feature_names_at_fit_time = loaded_model.get_feature_names_out()

data_test_with_features = pd.DataFrame(data_test, columns=feature_names_at_fit_time)

predictions_proba = loaded_model.predict_proba(data_test_with_features)  # Predict class probabilities
predictions_class = loaded_model.predict(data_test_with_features)

submission = pd.DataFrame(index=data_test.index, data=predictions_class, columns=['probability'])

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
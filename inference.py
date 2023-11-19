import pandas as pd
from data_preparation import data_prep
import joblib


PATH_DATA = "data"
MODEL_PATH = "model_2.pkl"

data_train, data_test, target = data_prep(PATH_DATA)

loaded_model = joblib.load(MODEL_PATH)

feature_names_at_fit_time = loaded_model.feature_names_in_

data_test_with_features = pd.DataFrame(data_test, columns=feature_names_at_fit_time)

predictions_proba = loaded_model.predict_proba(data_test_with_features)  # Predict class probabilities

print(predictions_proba[:, 0])

submission = pd.DataFrame(index=data_test.index, data=predictions_proba[:, 1], columns=['probability'])

submission.to_csv('result.csv')

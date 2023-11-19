from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import roc_auc_score
from data_preparation import data_prep

MODEL_PATH = "model_3.pkl"
PATH_DATA = "data"

#data = pd.read_csv('data_train_temp.csv', index_col='client_id')
#data_test = pd.read_csv('data_test_temp.csv', index_col='client_id')
#target = pd.read_csv('target_temp.csv', index_col='client_id')
data, data_test, target = data_prep(PATH_DATA)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=143)

model_rf = RandomForestClassifier(max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=500)
model_cb = CatBoostClassifier(depth=6, iterations=1500, l2_leaf_reg=4, learning_rate=0.05)
model_svc = SVC(probability=True, C=0.1, gamma= 'scale', kernel= 'linear')  # Note: probability=True for soft voting

voting_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', VotingClassifier(
        estimators=[
            ('RandomForest', model_rf),
            ('GradientBoosting', model_cb),
            ('SVM', model_svc)
        ],
        voting='soft'
    ))
])

voting_pipeline.fit(X_train, y_train)

y_pred = voting_pipeline.predict_proba(X_test)
roc_auc_test = roc_auc_score(y_test, y_pred[:, 1])

print("Ensemble Model Accuracy:", roc_auc_test)

joblib.dump(voting_pipeline, MODEL_PATH)


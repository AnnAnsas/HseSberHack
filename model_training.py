from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# from inference import MODEL_PATH, PATH_DATA
from data_preparation import data_prep

MODEL_PATH = "model_1.pkl"
PATH_DATA = "data"

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


data, data_test, target = data_prep(PATH_DATA)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Create pipelines for each model
cbc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', CatBoostClassifier())
])

xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier())
])


rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])


# Create a list of tuples where each tuple contains the model name and pipeline
models = [
    #('CatBoostClassifier', cbc_pipeline),
    ('XGBoostClassifier', xgb_pipeline),
    ('RandomForestClassifier', rf_pipeline)
]

# Define the hyperparameter grids for each model
param_grid_cbc = {
    'classifier__n_estimators': [100, 500, 1000],
    'classifier__max_depth': [1, 3, 7]
}

param_grid_xgb = {
    'classifier__n_estimators': [10, 50, 100],
    'classifier__max_depth': [1, 3, 7]
}


param_grid_rf = {
    'classifier__n_estimators': [100, 500, 1000],
    'classifier__max_depth': [1, 3, 7]
}

# Create a dictionary where keys are model names and values are corresponding hyperparameter grids
param_grids = {
    'CatBoostClassifier': param_grid_cbc,
    'XGBoostClassifier': param_grid_xgb,
    'RandomForestClassifier': param_grid_rf
}
# Perform grid search for each model
for model_name, pipeline in models:
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the corresponding ROC AUC score
    print(f"\nBest Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best ROC AUC Score for {model_name}: {grid_search.best_score_}")

    # Evaluate the best model on the test set
    y_pred = grid_search.predict(X_test)
    roc_auc_test = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC on Test Set for {model_name}: {roc_auc_test}")


# from sklearn.ensemble import VotingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# # Load a sample dataset (Breast Cancer dataset)
# data = load_breast_cancer()
# X, y = data.data, data.target
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Create individual models
# model_rf = RandomForestClassifier(random_state=42)
# model_gb = GradientBoostingClassifier(random_state=42)
# model_svc = SVC(probability=True, random_state=42)  # Note: probability=True for soft voting
#
# # Create a voting classifier
# voting_classifier = VotingClassifier(
#     estimators=[
#         ('RandomForest', model_rf),
#         ('GradientBoosting', model_gb),
#         ('SVM', model_svc)
#     ],
#     voting='soft'  # 'hard' for majority voting, 'soft' for weighted voting based on probabilities
# )
#
# # Fit the voting classifier on the training data
# voting_classifier.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = voting_classifier.predict(X_test)
#
# # Evaluate the accuracy of the ensemble model
# accuracy = accuracy_score(y_test, y_pred)
# print("Ensemble Model Accuracy:", accuracy)


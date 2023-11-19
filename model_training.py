from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


MODEL_PATH = "model_2.pkl"
PATH_DATA = "data"


data = pd.read_csv('data_train_temp.csv', index_col='client_id')
data_test = pd.read_csv('data_test_temp.csv', index_col='client_id')
target = pd.read_csv('target_temp.csv', index_col='client_id')
#data, data_test, target = data_prep(PATH_DATA)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=143)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', CatBoostClassifier(depth=6, iterations=250, l2_leaf_reg=4, learning_rate=0.05))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the model to a .pkl file
joblib.dump(pipeline, 'model_2.pkl')

#
# # Create pipelines for each model
# cbc_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', CatBoostClassifier())
# ])
#
# # Create a list of tuples where each tuple contains the model name and pipeline
# models = [
#     ('CatBoostClassifier', cbc_pipeline)
# ]
#
# # Define the hyperparameter grids for each model
# param_grid_cbc = {
#     'classifier__iterations': [150, 200, 250],
#     'classifier__learning_rate': [0.03, 0.05, 0.07],
#     'classifier__depth': [6, 7, 9],
#     'classifier__l2_leaf_reg': [2, 3, 4]
# }
#
# # Create a dictionary where keys are model names and values are corresponding hyperparameter grids
# param_grids = {
#     'CatBoostClassifier': param_grid_cbc
# }
# # Perform grid search for each model
# for model_name, pipeline in models:
#     param_grid = param_grids[model_name]
#     grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#
#     # Print the best parameters and the corresponding ROC AUC score
#     print(f"\nBest Parameters for {model_name}: {grid_search.best_params_}")
#     print(f"Best ROC AUC Score for {model_name}: {grid_search.best_score_}")
#
#     # Evaluate the best model on the test set
#     y_pred = grid_search.predict(X_test)
#     roc_auc_test = roc_auc_score(y_test, y_pred)
#     print(f"ROC AUC on Test Set for {model_name}: {roc_auc_test}")
#     joblib.dump(pipeline, 'model_.pkl')

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

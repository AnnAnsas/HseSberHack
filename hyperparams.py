from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

# from inference import MODEL_PATH, PATH_DATA
from data_preparation import data_prep

MODEL_PATH = "model_3.pkl"
PATH_DATA = "data"


data, data_test, target = data_prep(PATH_DATA)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


model_rf = RandomForestClassifier(random_state=42)
model_cb = CatBoostClassifier(depth=6, iterations=500, l2_leaf_reg=4, learning_rate=0.05)
model_svc = SVC(probability=True, random_state=42)

# Create pipelines for each model
cbc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', CatBoostClassifier())
])

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])


svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])


# Create a list of tuples where each tuple contains the model name and pipeline
models = [
    ('CatBoostClassifier', cbc_pipeline),
    ('RandomForestClassifier', rf_pipeline),
    ('SVClassifier', svc_pipeline),
]

# Define the hyperparameter grids for each model
param_grid_cbc = {
    'classifier__n_estimators': [800, 1000, 1500],
    'classifier__max_depth': [3,6,10]
}

param_grid_rf = {
    'classifier__n_estimators': [459, 500, 1000, 1500],
    'classifier__max_depth': [30, 100, 300, 500],
    'classifier__min_samples_split': [5],
    'classifier__min_samples_leaf': [2],
    'classifier__verbose': [2]
}

param_grid_cv = {
    'classifier__C': [0.1, 0.01, 0.05],
    'classifier__kernel': ['linear'],
    'classifier__gamma': ['scale'],
    'classifier__verbose': [True]
}

# Create a dictionary where keys are model names and values are corresponding hyperparameter grids
param_grids = {
    'RandomForestClassifier': param_grid_rf,
    'SVClassifier': param_grid_cv,
    'CatBoostClassifier': param_grid_cbc
}
file_path = "hypers.txt"

# Open the file in write mode ('w')
with open(file_path, 'w') as file:
    # Perform grid search for each model
    for model_name, pipeline in models:
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Print the best parameters and the corresponding ROC AUC score
        print(f"\nBest Parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best ROC AUC Score for {model_name}: {grid_search.best_score_}")
        file.write(f"\nBest Parameters for {model_name}: {grid_search.best_params_}")
        file.write(f"Best ROC AUC Score for {model_name}: {grid_search.best_score_}")

        # Evaluate the best model on the test set
        y_pred = grid_search.predict(X_test)
        roc_auc_test = roc_auc_score(y_test, y_pred)
        print(f"ROC AUC on Test Set for {model_name}: {roc_auc_test}")
        file.write(f"ROC AUC on Test Set for {model_name}: {roc_auc_test}")

        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv(f'grid_search_results_{model_name}.csv')

        # Visualize results (customize as needed)
        plt.figure(figsize=(10, 6))
        plt.plot(results['params'], results['mean_test_score'], label=f"Test Score for {model_name}")
        plt.plot(results['params'], results['mean_train_score'], label=f"Test Score for {model_name}")
        plt.xlabel('Number of Estimators')
        plt.ylabel('Score')
        plt.legend()
        plt.show()


print("Successful")

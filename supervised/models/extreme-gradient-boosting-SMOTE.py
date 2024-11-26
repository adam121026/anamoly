import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score
import xgboost as xgb
from itertools import product
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('../data/bank-additional-full_normalised.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

best_model_recall = None
best_model_f1 = None
best_recall = 0
best_f1 = 0
best_params_recall = None
best_params_f1 = None


# Iterate through all combinations
param_combinations = list(product(*param_grid.values()))
total = len(param_combinations)

for i, params in enumerate(param_combinations, 1):
    param = dict(zip(param_grid.keys(), params))
    model = xgb.XGBClassifier(
        eval_metric='mlogloss',
        **param
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"Progress: {i}/{total} - Evaluated params: {param}, Recall: {recall:.4f}, F1-score: {f1:.4f}", end='\r')
    if recall > best_recall:
        best_recall = recall
        best_model_recall = model
        best_params_recall = param
    if f1 > best_f1:
        best_f1 = f1
        best_model_f1 = model
        best_params_f1 = param

# Print the classification report for the best model
y_pred_recall = best_model_recall.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_recall))
print("Best parameters for recall:")
print(best_params_recall)

y_pred_f1 = best_model_f1.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_f1))
print("Best parameters for F1-score:")
print(best_params_f1)


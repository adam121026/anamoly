# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

# Load the dataset
data = pd.read_csv('../data/bank-additional-full_normalised.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom scorer for F1 score
scorer = make_scorer(f1_score, average='weighted')

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Create the Gradient Boosting model
model = GradientBoostingClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1, verbose=3)

# Train the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))
# Function to run the model and print the classification report
def run_model(X_train, y_train, X_test, y_test, sample_weight=None):
    grid_search.fit(X_train, y_train, sample_weight=sample_weight)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Run without weights
print("Running without weights:")
run_model(X_train, y_train, X_test, y_test)

# Run with sample weights
print("Running with sample weights:")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
run_model(X_train, y_train, X_test, y_test, sample_weight=sample_weights)

# Run with SMOTE
print("Running with SMOTE:")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
run_model(X_train_smote, y_train_smote, X_test, y_test)
# %%

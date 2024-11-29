# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('../data/bank-additional-full_normalised.csv')

# Assuming the target column is named 'class' and the rest are features
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

# Initialize Logistic Regression
logreg = LogisticRegression(max_iter=10000)

# Define the scorer
scorer = make_scorer(f1_score, pos_label=1)

# Initialize GridSearchCV with verbose
grid = GridSearchCV(logreg, param_grid, cv=5, scoring=scorer, verbose=1)

# Fit the model
grid.fit(X_train, y_train)

# Best parameters

# Make predictions
y_pred = grid.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Best Parameters:", grid.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# With class imbalance, we can use class_weight='balanced' to adjust the weights inversely proportional to class frequencies
logreg = LogisticRegression(max_iter=10000, class_weight='balanced')
grid = GridSearchCV(logreg, param_grid, cv=5, scoring=scorer, verbose=1)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print("Confusion Matrix (with class imbalance):")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (with class imbalance)')
plt.show()

print("Best Parameters:", grid.best_params_)
print("\nClassification Report (with class imbalance):")
print(classification_report(y_test, y_pred))

# %%

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, make_scorer, classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('../data/bank-additional-full_normalised.csv')

# Split the data into features and target
X = data.iloc[:, :-1]  # all columns except the last one
y = data.iloc[:, -1]   # the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Define the scoring criteria
scorer = make_scorer(recall_score, pos_label=1)

# Perform cross-validation and get the recall scores
recall_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=scorer)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Cross-validated Recall Scores: {recall_scores}')
print(f'Average Recall Score: {recall_scores.mean()}')
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


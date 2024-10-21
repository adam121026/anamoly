# %% Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# %% Load the data
df = pd.read_csv("../data/bank.csv")

# List of categorical columns to encode
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Label Encoding each categorical column
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])


# %% Separate the features from the target 'anamoly' column
features = df.drop(columns=['anamoly'])

# Scaling the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, df['anamoly'], test_size=0.3, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# %% Train the KNN model
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the predictions in the original DataFrame
df['predicted_anamoly_knn'] = knn.predict(features_scaled)

# Map 1 as 'yes' (anomaly) and 0 as 'no' (normal)
df['predicted_anamoly_knn'] = df['predicted_anamoly_knn'].map({1: 'yes', 0: 'no'})

# Save the updated data with KNN predictions to a CSV file
df.to_csv("../data/knn_anamoly_predictions.csv", columns=['anamoly', 'predicted_anamoly_knn'], index=False)


anomaly_distribution = df.groupby(['anamoly', 'predicted_anamoly_knn']).size().reset_index(name='count')

# Print the results
print(anomaly_distribution)
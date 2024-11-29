import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your dataset
df = pd.read_csv("../data/bank.csv")

# List of categorical columns to encode
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Label Encoding each categorical column
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Separate the features from the target 'anamoly' column
features = df.drop(columns=['anamoly'])

# Scaling the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['predicted_anamoly'] = iso_forest.fit_predict(features_scaled)

# Convert -1 (anamoly) to 1 and 1 (normal) to 0 for easier understanding
df['predicted_anamoly'] = df['predicted_anamoly'].apply(lambda x: 1 if x == -1 else 0)

# Mapping 1 as "yes" (anamoly) and 0 as "no" (normal)
df['predicted_anamoly'] = df['predicted_anamoly'].map({1: 'yes', 0: 'no'})

# Summary of the anamoly predictions ('yes' and 'no')
summary = df[['anamoly', 'predicted_anamoly']].value_counts()
print(summary)

# Save the updated data to a new CSV with both actual and predicted anomaly columns
df.to_csv("../data/anamoly_predictions.csv", columns=['anamoly', 'predicted_anamoly'], index=False)

# Display how 'yes' and 'no' are distributed in both original 'anamoly' and predicted 'predicted_anamoly'
anamoly_distribution = df.groupby(['anamoly', 'predicted_anamoly']).size().reset_index(name='count')
print(anamoly_distribution)

# %% Save the model
import joblib
import os

model_dir = "isolation-forest-models"
os.makedirs(model_dir, exist_ok=True)
model_list = os.listdir(model_dir)

joblib_file = f"{model_dir}/model_{len(model_list)}.joblib"
joblib.dump(iso_forest, joblib_file)

print(f"isolation-forest model saved to {joblib_file}")


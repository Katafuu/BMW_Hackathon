# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load training data
train_df = pd.read_csv('cleaned_train.csv')

# Preprocessing
target_columns = ['status_OK', 'physical_part_id']
X = train_df.drop(columns=target_columns)
y = train_df['status_OK']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y_encoded)

# Save the trained model, label encoder, and feature names
joblib.dump(model, 'trained_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
with open('trained_columns.txt', 'w') as f:
    f.write('\n'.join(X.columns))

print("Training completed and model saved.")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Load dataset
df = pd.read_csv("audio_features.csv", header=None)

if df.empty:
    raise ValueError("❌ Error: The dataset is empty. Check 'audio_features.csv'.")

# Separate features and labels
X = df.iloc[:, 1:].values  # Feature columns
y = df.iloc[:, 0].values   # Target labels

# Convert labels to numerical values
label_mapping = {"Human": 0, "Robot": 1}
y = np.array([label_mapping[label] for label in y])

# Ensure dataset has both classes
label_counts = Counter(y)
print("📊 Class distribution:", label_counts)

if len(label_counts) < 2:
    raise ValueError("❌ Error: Dataset contains only one class. Add more samples!")

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.2f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Human", "Robot"]))

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete! Saved as 'model.pkl' & 'scaler.pkl'.")

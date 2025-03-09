import os
import librosa
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(file_path):
    print(f"🔍 Processing {file_path}...")

    y, sr = librosa.load(file_path, sr=22050)

    if len(y) == 0:
        print(f"❌ Skipping empty file: {file_path}")
        return None

    # Extract 41 features (Ensure consistency with training)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)  # 20 features
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)  # 12 features
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)  # 7 features
    zero_crossing = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])  # 1 feature
    rms_energy = np.array([np.mean(librosa.feature.rms(y=y))])  # 1 feature

    # Combine into 41 features
    features = np.hstack([mfcc, chroma, spectral_contrast, zero_crossing, rms_energy])

    if features.shape[0] != 41:
        raise ValueError(f"❌ Feature mismatch! Expected 41 features, but got {features.shape[0]}.")

    return features

# Function to predict Human or Robot
def predict_voice(file_path):
    try:
        features = extract_features(file_path).reshape(1, -1)
        features = scaler.transform(features)  # Scale features
        prediction = model.predict(features)[0]
        return "Human" if prediction == 0 else "Robot"
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Function to evaluate model on test dataset
def evaluate_model(test_folder):
    actual_labels, predicted_labels = [], []

    for filename in os.listdir(test_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(test_folder, filename)
            actual_label = "Human" if "human" in filename.lower() else "Robot"
            prediction = predict_voice(file_path)

            if prediction:
                actual_labels.append(actual_label)
                predicted_labels.append(prediction)

    # Compute accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print(f"🎯 Model Accuracy on Test Dataset: {accuracy:.2f}")

    # Display classification report
    print("\n📊 Classification Report:")
    print(classification_report(actual_labels, predicted_labels, target_names=["Human", "Robot"]))

    # Save results
    with open("test_results.csv", "w") as f:
        f.write("Filename,Actual,Predicted\n")
        for filename, actual, predicted in zip(os.listdir(test_folder), actual_labels, predicted_labels):
            f.write(f"{filename},{actual},{predicted}\n")

    print("✅ Testing complete! Results saved in 'test_results.csv'.")

# Run evaluation on test dataset
evaluate_model("dataset/test_wav")

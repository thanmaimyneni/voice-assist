import librosa
import numpy as np
import os
import pandas as pd


def extract_features(file_path):
    print(f"🔍 Processing {file_path}...")

    y, sr = librosa.load(file_path, sr=22050)

    if len(y) == 0:
        print(f"❌ Skipping empty file: {file_path}")
        return None

    # Extract features (Ensure exactly 41 features)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)  # 20 features
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)  # 12 features
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)  # 7 features
    zero_crossing = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])  # 1 feature
    rms_energy = np.array([np.mean(librosa.feature.rms(y=y))])  # 1 feature ✅ (Previously missing)

    # Combine into 41 features
    features = np.hstack([mfcc, chroma, spectral_contrast, zero_crossing, rms_energy])

    print(f"✅ Extracted {features.shape[0]} features from {file_path}")
    return features


# Process dataset and save features
def process_dataset(directory, label):
    data = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        features = extract_features(file_path)
        if features is not None:
            data.append([label] + features.tolist())

    return data


# Paths to dataset
human_path = "dataset/human_wav"
robot_path = "dataset/robot_wav"

# Extract features
human_features = process_dataset(human_path, "Human")
robot_features = process_dataset(robot_path, "Robot")

# Save features to CSV
df = pd.DataFrame(human_features + robot_features)
df.to_csv("audio_features.csv", index=False, header=False)

print("✅ Feature extraction complete! Saved as 'audio_features.csv'")

import librosa

file_path = "dataset/test_wav/human_1.wav"  # Change to any test file

print(f"🔍 Trying to load {file_path}...")
y, sr = librosa.load(file_path, sr=22050)

if len(y) == 0:
    print("❌ Error: Audio file is empty or unreadable!")
else:
    print(f"✅ Successfully loaded {file_path}, Length: {len(y)} samples")

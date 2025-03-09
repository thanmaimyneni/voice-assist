import os
import pickle

# Get the absolute path to the model
model_path = os.path.abspath("../model.pkl")  # Adjust this if needed

if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found: {model_path}")

# Load the trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

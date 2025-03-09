# voice-assist
🔹 Project Overview
Your project is a machine learning-based system that detects whether a spoken voice is from a human or a bot (robotic voice). The system takes real-time voice input, processes its features, and classifies it using a trained model.

🔹 Key Features
✅ Real-time Voice Input - Users can speak directly into a microphone instead of uploading files.
✅ Feature Extraction - Extracts relevant features (MFCC, chroma, spectral contrast, etc.) from the audio.
✅ Machine Learning Model - A trained classifier (e.g., Random Forest, SVM) is used for prediction.
✅ Flask Web App - A simple web interface where users can click a button and start speaking.
✅ Prediction Output - Displays whether the detected voice is Human or Bot.

🔹 Technology Stack
🔹 Python - Core programming language
🔹 Flask - Backend framework for the web app
🔹 SoundDevice / PyAudio - Capturing real-time voice input
🔹 Librosa - Audio feature extraction
🔹 Scikit-learn - Model training and classification
🔹 HTML, JavaScript, CSS - Frontend for the web interface

🔹 Workflow of the Project
1️⃣ User clicks the "Start Recording" button on the web app.
2️⃣ Microphone records the voice input (3-5 seconds).
3️⃣ Feature extraction: MFCC, Chroma, Spectral Contrast, Zero-Crossing, RMS Energy.
4️⃣ Preprocessing: Features are normalized using StandardScaler.
5️⃣ Prediction: The trained machine learning model classifies the voice as Human or Bot.
6️⃣ Result Display: The web app shows the classification result.

🔹 Folder Structure
myproject/
├── dataset/                  # Contains training audio files
│   ├── human_wav/            # Human speech samples
│   ├── robot_wav/            # Robot-generated speech
├── models/                   # Trained models
│   ├── model.pkl             # Trained classifier
│   ├── scaler.pkl            # StandardScaler for feature scaling
├── webapp/                   # Flask web application
│   ├── uploads/              # Stores recorded audio files
│   ├── templates/            # Frontend HTML files
│   │   ├── index.html        # Web interface
│   ├── static/               # CSS, JS files
│   │   ├── styles.css        # Frontend styling
│   ├── app.py                # Flask backend for voice classification
├── extract_features.py        # Extracts audio features from dataset
├── train_model.py             # Trains the machine learning model
├── predict.py                 # Predicts Human vs. Bot for test audio
└── requirements.txt           # Dependencies for deployment

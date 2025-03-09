import os
import pyttsx3

# Define text file paths and output directories
data = {
    "human": ("dataset/human_text.txt", "dataset/human_wav/"),
    "robot": ("dataset/robot_text.txt", "dataset/robot_wav/")
}

# Initialize TTS engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')

def text_to_speech(label, input_file, output_folder, voice_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            output_file = os.path.join(output_folder, f"{label}_{i}.wav")
            engine.setProperty('voice', voices[voice_id].id)
            engine.save_to_file(line, output_file)

    engine.runAndWait()
    print(f"✅ WAV files generated in: {output_folder}")

# Convert both human and robot text to speech
text_to_speech("human", *data["human"], voice_id=0)  # Human voice
text_to_speech("robot", *data["robot"], voice_id=1)  # Robot voice

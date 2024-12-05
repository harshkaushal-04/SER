import sounddevice as sd
import librosa
import numpy as np
import os
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model # type: ignore
import datetime

# Load the trained model
model = load_model("best_model.keras")  # Ensure this model file exists
EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear']

# Directory to save recorded audio files
save_folder = "recorded_audio"
os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist

# Function to record audio
def record_audio(duration=3, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio.squeeze(), sr

# Function to save audio to a file
def save_audio(audio, sr):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_folder, f"recording_{timestamp}.wav")
    write(file_path, sr, (audio * 32767).astype(np.int16))  # Save as 16-bit PCM WAV
    print(f"Audio saved as: {file_path}")

# Function to extract MFCC features from audio
def extract_features_from_audio(audio, sr=22050, max_length=180000):
    if len(audio) > max_length:
        audio = audio[:max_length]
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Real-time prediction loop
while True:
    audio, sr = record_audio()
    
    # Replay recorded audio repeatedly until user stops
    while True:
        print("Playing back the recorded audio...")
        sd.play(audio, samplerate=sr)
        sd.wait()
        
        # Ask if the user wants to replay or stop
        print("Replay audio? (yes/no): ")
        if input().strip().lower() != 'yes':
            break

    # Save the recorded audio
    save_audio(audio, sr)

    # Predict emotion
    mfccs = extract_features_from_audio(audio, sr)
    prediction = model.predict(np.array([mfccs]))
    emotion = EMOTIONS[np.argmax(prediction)]
    print(f"Predicted Emotion: {emotion}")

    # Adaptive learning prompt
    print("Is the prediction correct? (yes/no): ")
    feedback = input().strip().lower()
    if feedback == 'no':
        print("What was the correct emotion? (happy/sad/angry/neutral/fear): ")
        correct_emotion = input().strip().lower()
        if correct_emotion in EMOTIONS:
            # Train the model with the new data point
            X_new = np.array([mfccs])
            y_new = np.array([EMOTIONS.index(correct_emotion)])
            model.fit(X_new, y_new, epochs=1, verbose=0)
            model.save("best_model.keras")  # Update the saved model
            print("Model updated with the correct label.")
        else:
            print("Invalid emotion entered.")
    elif feedback == 'yes':
        print("Great! No updates needed.")

    # Exit prompt
    print("Do you want to exit? (yes/no): ")
    if input().strip().lower() == 'yes':
        break

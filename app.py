from flask import Flask, request, jsonify, send_from_directory # type: ignore
import os
import numpy as np # type: ignore
import librosa # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import os
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

AUDIO_FOLDER = r"E:\SER_Project\recorded_audio"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list_files')
def list_files():
    files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
    return jsonify(files)


app = Flask(__name__)

# Define the folder to store recorded audio
AUDIO_FOLDER = r"E:\SER_Project\recorded_audio"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Load the pre-trained model
model = load_model("best_model.keras")  # Ensure this file exists
EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear']

# Function to extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    file_path = os.path.join(AUDIO_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    features = extract_features(file_path)
    prediction = model.predict(np.array([features]))
    emotion = EMOTIONS[np.argmax(prediction)]

    return jsonify({"emotion": emotion})

@app.route('/audio-files', methods=['GET'])
def list_audio_files():
    files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
    return jsonify(files)

@app.route('/recorded_audio/<filename>')
def get_audio_file(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)

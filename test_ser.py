import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score, classification_report

# Define emotion labels (update based on your model's training labels)
EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear']  # Adjust as needed

# Function to extract features (MFCCs) from audio files
def extract_features(file_path, max_len=130):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        return mfccs.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load the trained model
model = load_model("best_model.keras")  # Update with the path to your saved model

# Path to the dataset directory
data_dir = "E:\SER_Project\Datasets\happy\03-01-01-01-01-01-04.wav"  # Replace with the actual dataset path

# Collect files and labels
audio_files = []
labels = []

# Emotion label mapping (adjust for your dataset structure)
emotion_map = {'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3, 'fear': 4}  # Update as needed

# Process each audio file
for subdir, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            label = next((emotion_map[key] for key in emotion_map if key in file.lower()), None)
            if label is not None:
                features = extract_features(file_path)
                if features is not None:
                    audio_files.append(features)
                    labels.append(label)

# Convert lists to numpy arrays
X_test = np.array(audio_files)
y_test = np.array(labels)

# Ensure data is valid before prediction
if X_test.shape[0] == 0:
    raise ValueError("No valid audio files processed. Check your dataset path and format.")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy on {data_dir}: {accuracy:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=EMOTIONS))


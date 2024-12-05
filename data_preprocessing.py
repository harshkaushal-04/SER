import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# Set the path to your organized datasets folder
DATASET_PATH = "E:\SER_Project\Datasets"  # Make sure to adjust the path if necessary

# Define the emotions as labels (adjust this if necessary)
EMOTIONS = {'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3, 'fear': 4}

def extract_features(file_path, max_length=180000):
    # Load the audio file
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    
    # Ensure the length of the audio doesn't exceed the max length
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    # Extract MFCC features from the audio file
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    # Take the mean of the MFCCs to reduce dimensionality
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Prepare the features and labels
features, labels = [], []
for emotion, label in EMOTIONS.items():
    emotion_folder = os.path.join(DATASET_PATH, emotion)
    for file in os.listdir(emotion_folder):
        file_path = os.path.join(emotion_folder, file)
        try:
            # Extract features and append them to the list
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Convert features and labels into numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Save the preprocessed data for future use
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Data preprocessing complete. Files saved as .npy.")

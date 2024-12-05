import os
import shutil

# Paths to the datasets (update these paths)
datasets = {
    "SAVEE": "E:\SER_Project\Datasets\SAVEE",          # Path to SAVEE dataset folder
    "TESS": "E:\SER_Project\Datasets\TESS",            # Path to TESS dataset folder
    "CREMA-D": "E:\SER_Project\Datasets\CREMA-D",      # Path to CREMA-D dataset folder
    "RAVDESS": "E:\SER_Project\Datasets\RAVDESS"       # Path to RAVDESS dataset folder
}

# Target folder for organized data
dest_path = "E:\SER_Project\Datasets"

# Mapping emotions to their labels in each dataset
emotion_map = {
    "happy": {
        "SAVEE": ["h"],               # Prefix in SAVEE
        "TESS": ["Happy"],            # Folder name in TESS
        "CREMA-D": ["HAP"],           # Label in CREMA-D
        "RAVDESS": ["03"]             # Code in RAVDESS
    },
    "sad": {
        "SAVEE": ["sa"],              # Prefix in SAVEE
        "TESS": ["Sad"],              # Folder name in TESS
        "CREMA-D": ["SAD"],           # Label in CREMA-D
        "RAVDESS": ["04"]             # Code in RAVDESS
    },
    "angry": {
        "SAVEE": ["a"],               # Prefix in SAVEE
        "TESS": ["Angry"],            # Folder name in TESS
        "CREMA-D": ["ANG"],           # Label in CREMA-D
        "RAVDESS": ["05"]             # Code in RAVDESS
    },
    "neutral": {
        "SAVEE": ["n"],               # Prefix in SAVEE
        "TESS": [],                   # TESS does not have neutral
        "CREMA-D": ["NEU"],           # Label in CREMA-D
        "RAVDESS": ["01", "02"]       # Codes in RAVDESS (neutral, calm)
    },
    "fear": {
        "SAVEE": ["f"],               # Prefix in SAVEE
        "TESS": ["Fear"],             # Folder name in TESS
        "CREMA-D": ["FEA"],           # Label in CREMA-D
        "RAVDESS": ["06"]             # Code in RAVDESS
    }
}

# Function to move a file to its respective emotion folder
def move_file(file_path, emotion):
    target_folder = os.path.join(dest_path, emotion)
    os.makedirs(target_folder, exist_ok=True)  # Ensure the folder exists

    # Resolve filename conflicts
    destination_file = os.path.join(target_folder, os.path.basename(file_path))
    if os.path.exists(destination_file):
        base, ext = os.path.splitext(destination_file)
        counter = 1
        # Append a number to the filename until it's unique
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        destination_file = f"{base}_{counter}{ext}"

    shutil.move(file_path, destination_file)


# Process each dataset
for dataset_name, dataset_path in datasets.items():
    print(f"Processing dataset: {dataset_name}")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            for emotion, labels in emotion_map.items():
                # Get the labels specific to this dataset
                dataset_labels = labels.get(dataset_name, [])
                # Check if the file matches any label
                if any(label in file for label in dataset_labels):
                    move_file(file_path, emotion)
                    print(f"Moved {file} to {emotion}")
                    break

print("File organization complete!")

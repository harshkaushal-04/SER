import os
import zipfile
import subprocess

# Dataset names and their Kaggle URLs
datasets = {
    "CREMA-D": "ejlok1/cremad",
    "SAVEE": "ejlok1/surrey-audiovisual-expressed-emotion-savee"
}

# Base path for the project
base_path = "./SER_Project/Datasets"

# Create the base directory
os.makedirs(base_path, exist_ok=True)

def download_and_unzip(dataset_name, kaggle_url):
    dataset_path = os.path.join(base_path, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    # Download the dataset
    print(f"Downloading {dataset_name}...")
    subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_url, "-p", dataset_path])

    # Find and unzip the downloaded file
    for file in os.listdir(dataset_path):
        if file.endswith(".zip"):
            zip_file_path = os.path.join(dataset_path, file)
            print(f"Unzipping {file}...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            os.remove(zip_file_path)  # Remove the zip file after extraction
            print(f"{dataset_name} unzipped successfully!")

# Download and unzip each dataset
for name, url in datasets.items():
    download_and_unzip(name, url)

print("All datasets downloaded and unzipped!")

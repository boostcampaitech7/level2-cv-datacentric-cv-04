import kagglehub
import shutil
import os

# Define the target path for the downloaded dataset
target_path = os.path.join(os.path.dirname(__file__), "data", "sroie_receipt")

# Ensure the target directory exists
if not os.path.exists(target_path):
    print(f"Creating target directory at {target_path}...")
    os.makedirs(target_path)

# Download the latest version of the SROIE dataset
download_path = kagglehub.dataset_download("urbikn/sroie-datasetv2")

print("Path to downloaded dataset files:", download_path)

# Ensure the target directory exists
os.makedirs(target_path, exist_ok=True)

# Check if the downloaded content is a directory
downloaded_content = os.path.join(download_path, "SROIE2019")
if os.path.isdir(downloaded_content):
    # Move the entire contents of SROIE2019 to the target directory
    for item in os.listdir(downloaded_content):
        source_item = os.path.join(downloaded_content, item)
        target_item = os.path.join(target_path, item)
        if os.path.isdir(source_item):
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
        else:
            shutil.copy2(source_item, target_item)
    print(f"Moved contents of {downloaded_content} to {target_path}")
else:
    print("Expected a directory named SROIE2019 but did not find it.")

print("Dataset setup is complete!")
# from datasets import load_dataset

# ds = load_dataset("scene_parse_150", split="train[:50]")

# ds = ds.train_test_split(test_size=0.2)
# train_ds = ds["train"]
# test_ds = ds["test"]
# train_ds[0]

# train_ds[0]["image"]

# import json
# from huggingface_hub import cached_download, hf_hub_url

# repo_id = "huggingface/label-files"
# filename = "ade20k-id2label.json"
# id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
# id2label = {int(k): v for k, v in id2label.items()}
# label2id = {v: k for k, v in id2label.items()}
# num_labels = len(id2label)
import os
import shutil
import random

# Path to the dataset directory
dataset_dir = "./Dataset"

# Path to the new valdataset directory
valdataset_dir = "./ValDataset"

# Function to create directory recursively
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create valdataset directories to mirror the dataset structure
create_directory(valdataset_dir)
create_directory(os.path.join(valdataset_dir, "VineNet"))
create_directory(os.path.join(valdataset_dir, "VineNet/images"))
create_directory(os.path.join(valdataset_dir, "VineNet/masks"))

# Get list of image files
image_files = sorted([f for f in os.listdir(os.path.join(dataset_dir, "VineNet/images")) if f.endswith(".png")])

# Shuffle the list of image files
random.shuffle(image_files)

# Select the first 100 image files
selected_image_files = image_files[:100]

# Copy selected image files and their corresponding masks to valdataset
for image_file in selected_image_files:
    # Get corresponding mask file
    mask_file = image_file.replace(".png", "_instanceIds.png")
    
    # Source paths
    src_image_path = os.path.join(dataset_dir, "VineNet/images", image_file)
    src_mask_path = os.path.join(dataset_dir, "VineNet/masks", mask_file)
    
    # Destination paths
    dest_image_path = os.path.join(valdataset_dir, "VineNet/images", image_file)
    dest_mask_path = os.path.join(valdataset_dir, "VineNet/masks", mask_file)
    
    # Move image file
    os.rename(src_image_path, dest_image_path)
    
    # Move mask file
    os.rename(src_mask_path, dest_mask_path)

print("Extraction completed.")


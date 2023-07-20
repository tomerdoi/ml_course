import glob
import os
import requests
import tarfile
import shutil
import pandas as pd
import random
import yaml
from pathlib import Path
import scipy.io
import torch

# Set a random seed for reproducibility
random.seed(42)


# Function to download and extract the dataset and labels
def download_and_extract_dataset(url, save_dir):
    # Create the save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Download the dataset file
    dataset_file = os.path.join(save_dir, url.split('/')[-1])
    if not os.path.exists(dataset_file):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(dataset_file, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        del response

    # Extract the dataset file
    print("Extracting dataset...")
    with tarfile.open(dataset_file, 'r:gz') as tar:
        tar.extractall(save_dir)

    print("Dataset downloaded and extracted successfully.")


# Function to split the dataset into training, validation, and test sets
def split_dataset(data_root, split_ratio=(0.5, 0.25, 0.25), num_runs=2):
    all_image_paths = list(Path(data_root).glob('**/*.jpg'))
    random.shuffle(all_image_paths)

    num_images = len(all_image_paths)
    train_size = int(num_images * split_ratio[0])
    val_size = int(num_images * split_ratio[1])
    test_size = num_images - train_size - val_size

    for run in range(num_runs):
        random.shuffle(all_image_paths)

        # Create directories for each split
        train_dir = Path(f"/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/run_{run + 1}/train")
        val_dir = Path(f"/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/run_{run + 1}/val")
        test_dir = Path(f"/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/run_{run + 1}/test")
        if os.path.isdir(train_dir):
            shutil.rmtree(train_dir)
        if os.path.isdir(val_dir):
            shutil.rmtree(val_dir)
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Move images to their respective directories
        for i in range(train_size):
            shutil.copy(all_image_paths[i], train_dir)

        for i in range(train_size, train_size + val_size):
            shutil.copy(all_image_paths[i], val_dir)

        for i in range(train_size + val_size, num_images):
            shutil.copy(all_image_paths[i], test_dir)


# Download and extract the dataset
dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
# Replace with the desired save directory
dataset_save_dir = "/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/dataset"
data_root = os.path.join(dataset_save_dir, "jpg")
download_and_extract_dataset(dataset_url, dataset_save_dir)

# Download the labels file
labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
labels_save_path = os.path.join(dataset_save_dir, "imagelabels.mat")
if not os.path.exists(labels_save_path):
    print(f"Downloading labels from {labels_url}...")
    response = requests.get(labels_url, stream=True)
    with open(labels_save_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    del response
print("Labels downloaded successfully.")

# Load the labels from the .mat file
labels_data = scipy.io.loadmat(labels_save_path)
labels = labels_data['labels'][0]


def generate_labels_cache(dataset_root, labels):
    labels_cache_path = os.path.join(dataset_root, "labels_cache.csv")
    images = list(sorted([os.path.join(data_root, image_path) for image_path in glob.glob(data_root + '/*.jpg')]))
    labels_df = pd.DataFrame({"image": images, "label": labels})
    labels_df.to_csv(labels_cache_path, index=False)


def train_yolov5_flowers(run):
    os.chdir('./yolov5')  # Replace with the path to the YOLOv5 repository

    # Split the dataset into training, validation, and test sets
    data_root = os.path.join(dataset_save_dir, f"run_{run}")
    split_dataset(data_root)

    # Convert the labels to integers
    labels = labels_data['labels'][0].tolist()

    # Generate labels cache file
    for phase in ['train', 'val']:
        generate_labels_cache(os.path.join(data_root, phase), labels)

    # Create the flowers.yaml file for dataset configuration
    with open('data/flowers.yaml', 'w') as f:
        yaml_dict = {
            'train': f"/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/run_{run}/train",
            'val': f"/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/run_{run}/val",
            'nc': 102,  # Number of classes (102 flower categories)
            'names': ['class_' + str(i) for i in range(102)]  # List of class names (optional)
        }
        yaml.dump(yaml_dict, f)

    # Start training
    os.system(f"python train.py --img-size 640 --batch-size 16 --epochs 30 --data data/flowers.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name flowers_run_{run}")

    # Move the training results to the appropriate directory
    output_dir = f"/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/run_{run}"
    os.makedirs(output_dir, exist_ok=True)
    os.replace('runs/train/flowers_run_{run}', output_dir)


# Split the dataset into training, validation, and test sets
split_dataset(data_root)

# Train the models for each run
for run in range(1, 3):
    train_yolov5_flowers(run)

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
from logger_utils import LoggerUtils

# Set a random seed for reproducibility
random.seed(42)


class YoloOOB:
    def __init__(self):
        self.train_folder = '/Users/tomerdoitshman/Desktop/other/D_non_shared/ass3_dataset'
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='yolov5oob.log')

# Function to download and extract the dataset and labels
    def download_and_extract_dataset(self, url, save_dir):
        try:
            # Create the save directory if it doesn't exist
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            # Download the dataset file
            dataset_file = os.path.join(save_dir, url.split('/')[-1])
            if not os.path.exists(dataset_file):
                self.logger.info(f"Downloading dataset from {url}...")
                response = requests.get(url, stream=True)
                with open(dataset_file, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                del response
            # Extract the dataset file
            self.logger.info("Extracting dataset...")
            with tarfile.open(dataset_file, 'r:gz') as tar:
                tar.extractall(save_dir)

            self.logger.info("Dataset downloaded and extracted successfully.")
        except Exception as e:
            self.logger.error('Exception %s occurred during download_and_extract_dataset.' % e)


    # Function to split the dataset into training, validation, and test sets
    def split_dataset(self, data_root, split_ratio=(0.5, 0.25, 0.25), num_runs=2):
        try:
            all_image_paths = list(Path('/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/dataset').glob('**/*.jpg'))
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
        except Exception as e:
            self.logger.error('Exception %s occurred during split_dataset.' % e)

    def generate_labels_cache(self, dataset_root, labels):
        try:
            labels_cache_path = os.path.join(dataset_root, "train.cache")
            images = list(sorted([os.path.join(data_root, image_path) for image_path in glob.glob(data_root + '/*.jpg')]))
            labels_df = pd.DataFrame({"image": images, "label": labels})
            labels_df.to_csv(labels_cache_path, index=False)
        except Exception as e:
            self.logger.error('Exception %s occurred during generate_labels_cache.' % e)

    def train_yolov5_flowers(self, run):
        try:
            os.chdir('./yolov5')  # Replace with the path to the YOLOv5 repository
            # Split the dataset into training, validation, and test sets
            run_outputs_path = os.path.join(
                '/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/', f"run_{run}")
            self.split_dataset(data_root)
            # Convert the labels to integers
            labels = labels_data['labels'][0].tolist()
            # Generate labels cache file
            for phase in ['train', 'val']:
                self.generate_labels_cache(run_outputs_path, labels)
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
            os.system(
                f"python train.py --img-size 640 --batch-size 16 --epochs 30 --data data/flowers.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name flowers_run_{run}")
            # Move the training results to the appropriate directory
            output_dir = f"/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/outputs/yolo_outputs/run_{run}"
            os.makedirs(output_dir, exist_ok=True)
            os.replace('runs/train/flowers_run_{run}', output_dir)
        except Exception as e:
            self.logger.error('Exception %s occurred during train_yolov5_flowers.' % e)

    def run(self):
        try:
            # Download and extract the dataset
            dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
            # Replace with the desired save directory
            dataset_save_dir = "/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/dataset"
            data_root = os.path.join(dataset_save_dir, "jpg")
            self.download_and_extract_dataset(dataset_url, dataset_save_dir)
            # Download the labels file
            labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
            labels_save_path = os.path.join(dataset_save_dir, "imagelabels.mat")
            if not os.path.exists(labels_save_path):
                self.logger.info(f"Downloading labels from {labels_url}...")
                response = requests.get(labels_url, stream=True)
                with open(labels_save_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                del response
            self.logger.info("Labels downloaded successfully.")
            # Load the labels from the .mat file
            labels_data = scipy.io.loadmat(labels_save_path)
            labels = labels_data['labels'][0]
            # Split the dataset into training, validation, and test sets
            self.split_dataset(data_root)
            # Train the models for each run
            for run in range(1, 3):
                self.train_yolov5_flowers(run)
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)


if __name__ == '__main__':
    yolo_oob = YoloOOB()
    yolo_oob.run()

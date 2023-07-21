import glob
import requests
import tarfile
import shutil
import random
import yaml
import os
import scipy.io
from PIL import Image  # Import the Image class from PIL (or Pillow)
from pathlib import Path
import scipy.io
from logger_utils import LoggerUtils


# change this PROJECT_PATH according to your local env
PROJECT_PATH = '/Users/tomerdoitshman/PycharmProjects/ml_course/assignment3/'

DATASETS_PATH = os.path.join(PROJECT_PATH, 'datasets')
FLOWERS_PATH = os.path.join(DATASETS_PATH, 'flowers')
RUN_1_PATH = os.path.join(FLOWERS_PATH, 'run_1')
RUN_2_PATH = os.path.join(FLOWERS_PATH, 'run_2')
ALL_IMAGES_PATH = os.path.join(DATASETS_PATH, 'jpg')
ALL_LABELS_PATH = os.path.join(FLOWERS_PATH, 'labels')
ALL_LABELS_MAT_PATH = os.path.join(DATASETS_PATH, 'imagelabels.mat')
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"


# Set a random seed for reproducibility
random.seed(42)


class YoloOOB:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='yolov5oob.log')

# Function to download and extract the dataset and labels
    def download_and_extract_dataset(self):
        try:
            # Create the save directory if it doesn't exist
            Path(DATASETS_PATH).mkdir(parents=True, exist_ok=True)
            # Download the dataset file
            dataset_file = os.path.join(DATASETS_PATH, DATASET_URL.split('/')[-1])
            if not os.path.exists(dataset_file):
                self.logger.info(f"Downloading dataset from {DATASET_URL}...")
                response = requests.get(DATASET_URL, stream=True)
                with open(dataset_file, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                del response
            # Extract the dataset file
            self.logger.info("Extracting dataset...")
            with tarfile.open(dataset_file, 'r:gz') as tar:
                tar.extractall(DATASETS_PATH)
            self.logger.info("Dataset downloaded and extracted successfully.")
            # Download the labels file
            labels_save_path = os.path.join(DATASETS_PATH, "imagelabels.mat")
            if not os.path.exists(labels_save_path):
                self.logger.info(f"Downloading labels from {LABELS_URL}...")
                response = requests.get(LABELS_URL, stream=True)
                with open(labels_save_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                del response
            self.logger.info("Labels downloaded successfully.")
        except Exception as e:
            self.logger.error('Exception %s occurred during download_and_extract_dataset.' % e)
    
    def create_dirs(self, run):
        try:
            # Create directories for each split
            # create images dirs
            train_dir_images = Path(FLOWERS_PATH + f"/run_{run + 1}/train/images")
            val_dir_images = Path(FLOWERS_PATH + f"/run_{run + 1}/val/images")
            test_dir_images = Path(FLOWERS_PATH + f"/run_{run + 1}/test/images")
            if os.path.isdir(train_dir_images):
                shutil.rmtree(train_dir_images)
            if os.path.isdir(val_dir_images):
                shutil.rmtree(val_dir_images)
            if os.path.isdir(test_dir_images):
                shutil.rmtree(test_dir_images)
            train_dir_images.mkdir(parents=True, exist_ok=True)
            val_dir_images.mkdir(parents=True, exist_ok=True)
            test_dir_images.mkdir(parents=True, exist_ok=True)
            # create labels dirs
            train_dir_labels = Path(FLOWERS_PATH + f"/run_{run + 1}/train/labels")
            val_dir_labels = Path(FLOWERS_PATH + f"/run_{run + 1}/val/labels")
            test_dir_labels = Path(FLOWERS_PATH + f"/run_{run + 1}/test/labels")
            if os.path.isdir(train_dir_labels):
                shutil.rmtree(train_dir_labels)
            if os.path.isdir(val_dir_labels):
                shutil.rmtree(val_dir_labels)
            if os.path.isdir(test_dir_labels):
                shutil.rmtree(test_dir_labels)
            train_dir_labels.mkdir(parents=True, exist_ok=True)
            val_dir_labels.mkdir(parents=True, exist_ok=True)
            test_dir_labels.mkdir(parents=True, exist_ok=True)
            return train_dir_images, val_dir_images, test_dir_images, train_dir_labels, val_dir_labels, test_dir_labels
        except Exception as e:
            self.logger.error('Exception %s occurred during create_dirs.' % e)
            
    def split_dataset(self, labels, split_ratio=(0.5, 0.25, 0.25), num_runs=2):
        try:
            all_image_paths = glob.glob(ALL_IMAGES_PATH + '/*.jpg')
            num_images = len(all_image_paths)
            train_size = int(num_images * split_ratio[0])
            val_size = int(num_images * split_ratio[1])
            test_size = num_images - train_size - val_size
            for run in range(num_runs):
                random.shuffle(all_image_paths)
                train_dir_images, val_dir_images, test_dir_images, train_dir_labels,\
                    val_dir_labels, test_dir_labels = self.create_dirs(run)
                # Move images to their respective directories
                for i in range(train_size):
                    shutil.copy(all_image_paths[i], train_dir_images)
                    self.convert_label_to_yolo_format(all_image_paths[i], labels[i], run + 1, 'train')
                for i in range(train_size, train_size + val_size):
                    shutil.copy(all_image_paths[i], val_dir_images)
                    self.convert_label_to_yolo_format(all_image_paths[i], labels[i], run + 1, 'val')
                for i in range(train_size + val_size, num_images):
                    shutil.copy(all_image_paths[i], test_dir_images)
                    self.convert_label_to_yolo_format(all_image_paths[i], labels[i], run + 1, 'test')
        except Exception as e:
            self.logger.error('Exception %s occurred during split_dataset.' % e)

    def train_yolov5_flowers(self, run):
        try:
            os.chdir('./yolov5')  # Replace with the path to the YOLOv5 repository
            with open('data/flowers.yaml', 'w') as f:
                yaml_dict = {
                    'path': '../datasets/flowers',
                    'train': f"run_{run}/train",
                    'val': f"run_{run}/val",
                    'nc': 102,  # Number of classes (102 flower categories)
                    'names': [str(i) for i in range(102)]  # List of class names (optional)
                }
                yaml.dump(yaml_dict, f)
            # Start training
            cmd_command = f"python train.py --img-size 640 --batch-size 16 --epochs 30 --data data/flowers.yaml --cfg" \
                          f" models/yolov5s.yaml --weights yolov5s.pt --name flowers_run_{run}"
            os.system(cmd_command)
        except Exception as e:
            self.logger.error('Exception %s occurred during train_yolov5_flowers.' % e)

    def load_labels(self):
        try:
            # Load the labels from the .mat file
            labels_data = scipy.io.loadmat(ALL_LABELS_MAT_PATH)
            labels = labels_data['labels'][0]
            return labels
        except Exception as e:
            self.logger.error('Exception %s occurred during load_labels.' % e)

    def run(self):
        try:
            self.download_and_extract_dataset()
            labels = self.load_labels()
            self.split_dataset(labels=labels)
            for run in range(1, 3):
                self.train_yolov5_flowers(run)
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)

    def convert_label_to_yolo_format(self, image_path, label, run_num, set_name):
        try:
            class_index = label - 1  # Assuming class labels in the dataset are 1-indexed, while YOLOv5
            # requires 0-indexed
            image = Image.open(image_path)
            image_width, image_height = image.size
            # Assuming you have the bounding box coordinates of the flower in the format [x_min, y_min,
            # x_max, y_max]
            # If you don't have the bounding box coordinates, you'll need to obtain them from the
            # dataset or annotations.
            x_min, y_min, x_max, y_max = [0.0, 0.0, 1.0, 1.0]  # Replace with actual bounding box coordinates
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            annotation = f"{class_index} {x_center} {y_center} {width} {height}"
            labels_dir_path = os.path.join(FLOWERS_PATH, 'run_%d' % run_num, set_name, 'labels')
            with open(os.path.join(labels_dir_path, image_path.split('/')[-1].replace('.jpg', '.txt')), 'w') as fp:
                fp.write(annotation)
        except Exception as e:
            self.logger.error('Exception %s occurred during convert_label_to_yolo_format.' % e)


if __name__ == '__main__':
    yolo_oob = YoloOOB()
    yolo_oob.run()

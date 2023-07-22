import os
import shutil
import tarfile
from io import BytesIO

import numpy as np
import requests
import scipy.io
from sklearn.model_selection import train_test_split


def download_images(output_dir='.'):
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    file_res = requests.get(images_url)
    file_obj = BytesIO(file_res.content)

    with tarfile.open(fileobj=file_obj) as f:
        f.extractall(output_dir)


def get_labels():
    labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
    file_res = requests.get(labels_url)
    file_obj = BytesIO(file_res.content)

    return scipy.io.loadmat(file_obj)['labels'].flatten()


def get_default_split():
    labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
    file_res = requests.get(labels_url)
    file_obj = BytesIO(file_res.content)

    splits = scipy.io.loadmat(file_obj)
    train_set_ids = splits['trnid']
    val_set_ids = splits['valid']
    test_set_ids = splits['tstid']

    return train_set_ids, val_set_ids, test_set_ids


def random_split(data_ids, train_sz=0.5, val_sz=0.25):
    train_set_ids, test_set_ids = train_test_split(data_ids, train_size=train_sz)
    val_set_ids, test_set_ids = train_test_split(test_set_ids, train_size=val_sz / (1 - train_sz))

    train_set_ids = np.array(train_set_ids).reshape((1, -1))
    val_set_ids = np.array(val_set_ids).reshape((1, -1))
    test_set_ids = np.array(test_set_ids).reshape((1, -1))

    return train_set_ids, val_set_ids, test_set_ids


def format_folder(images_folder, output_folder, train_set, val_set, test_set, labels):
    for set_, ids in {'train': train_set, 'val': val_set, 'test': test_set}.items():
        os.makedirs(os.path.join(output_folder, set_), exist_ok=True)
        for id_ in ids[0]:
            file_name = f'image_{id_:05}.jpg'
            label = labels[id_ - 1]
            if not os.path.exists(os.path.join(output_folder, set_, str(label))):
                os.mkdir(os.path.join(output_folder, set_, str(label)))
            shutil.copy(os.path.join(images_folder, file_name), os.path.join(output_folder, set_, str(label), file_name))

import numpy as np
import csv
import os
import glob
from scipy.ndimage import imread
import yaml
from collections import OrderedDict


def _load_images(tiny_imagenet_base_path, val_only=False):

    # Load dataset
    ds_base_path = tiny_imagenet_base_path

    with open(os.path.join(ds_base_path, 'wnids.txt'), 'rt') as f:
        names = f.readlines()

    class_name_id_dict = {name.strip(): clsid for clsid, name in enumerate(names)}

    print("Loading validation data...")
    X_val = []
    y_val = []
    csv_path = os.path.join(ds_base_path, "val", "val_annotations.txt")
    with open(csv_path, 'rt') as csv_file:
        for row in csv.reader(csv_file, delimiter='\t'):
            image_path = os.path.join(ds_base_path, 'val', 'images', row[0])
            class_name = row[1]

            image_data = imread(image_path)  # shape=(h,w,c)
            if len(image_data.shape) == 2:
                image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)
            X_val.append(image_data)
            y_val.append(class_name_id_dict[class_name])

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    if val_only:
        return X_val, y_val

    print("Loading training data...")
    X_train = []
    y_train = []
    for class_name, clsid in class_name_id_dict.items():
        print("Class {}...".format(class_name))
        for image_path in glob.glob(os.path.join(ds_base_path, 'train', class_name, 'images', '*')):
            image_data = imread(image_path)  # shape=(h,w,c)
            if len(image_data.shape) == 2:
                image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)
            X_train.append(image_data)
            y_train.append(clsid)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train, X_val, y_val


def _ds_stats(X, y):
    # Return a string containing some stats on the dataset.
    unique, counts = np.unique(np.array(y), return_counts=True)
    return str(dict(zip(unique, counts)))


def load_dataset(tiny_imagenet_base_path, ds_cache_path=None):
    """
    Loads the Tiny ImageNet dataset (Training and Validation sets). Thankfully, this dataset fits in memory all at once.
    :param tiny_imagenet_base_path: base path of Tiny ImageNet.
    :param ds_cache_path: If provided, caches the dataset to this NPZ file. If it already exists, the dataset is loaded from there.
    :return: X_train, y_train, X_val, y_val. X has format (n,h,w,c)
    """

    # Cache the preprocessed dataset, so we can start up faster. It's small enough so we can do this easily.
    if ds_cache_path is not None and os.path.isfile(ds_cache_path):
        print('Loading preprocessed dataset from "{}"...'.format(ds_cache_path))
        loaded_ds = np.load(ds_cache_path, encoding='latin1')['dataset']
    else:
        print('Loading dataset from source images...')
        loaded_ds = _load_images(tiny_imagenet_base_path)

        if ds_cache_path is not None:
            # TODO: this is broken since Python 3.6! Can't store a tuple of heterogenous arrays anympre.
            #  Fix to handle the same way as load_dataset_val()!
            np.savez_compressed(ds_cache_path, dataset=loaded_ds)
            print('Saved preprocessed dataset to "{}".'.format(ds_cache_path))

    X_train, y_train, X_val, y_val = loaded_ds

    # print('Loaded dataset: n_train={}; n_test={}'.format(len(X_train), len(X_val)))
    # print('Number of examples per class:')
    # print('\tTrain: {}'.format(_ds_stats(X_train, y_train)))
    # print('\tVal: {}'.format(_ds_stats(X_val, y_val)))
    # print('All image data has shape {}'.format(X_train.shape[1:]))

    return X_train, y_train, X_val, y_val


def load_dataset_val_only(tiny_imagenet_base_path, ds_val_cache_path=None):
    """
    Loads the Tiny ImageNet dataset (Validation set only). For speedy evaluation - is faster than load_dataset().
    :param tiny_imagenet_base_path: base path of Tiny ImageNet.
    :param ds_val_cache_path: If provided, caches the dataset to this NPZ file. If it already exists, the dataset is loaded from there.
    :return: X_val, y_val. X has format (n,h,w,c)
    """

    # Cache the preprocessed dataset, so we can start up faster. It's small enough so we can do this easily.
    if ds_val_cache_path is not None and os.path.isfile(ds_val_cache_path):
        print('Loading preprocessed dataset from "{}"...'.format(ds_val_cache_path))
        loaded_ds = np.load(ds_val_cache_path, encoding='latin1')
        X_val = loaded_ds['X_val']
        y_val = loaded_ds['y_val']
    else:
        print('Loading dataset from source images...')
        X_val, y_val = _load_images(tiny_imagenet_base_path, val_only=True)

        if ds_val_cache_path is not None:
            np.savez_compressed(ds_val_cache_path, X_val=X_val, y_val=y_val)
            print('Saved preprocessed dataset to "{}".'.format(ds_val_cache_path))

    return X_val, y_val


def load_test_imgs():
    """
    Loads the AVC test images from the AVC package.
    :return: X_test, y_test. X has format (n,h,w,c)
    """

    import adversarial_vision_challenge
    avc_test_path = os.path.join(os.path.dirname(adversarial_vision_challenge.__file__), "test_images")

    with open(os.path.join(avc_test_path, "labels.yml"), "rt") as f:
        labels = yaml.load(f)

    # Important! Need to load the images in the same order every time!
    ordered = OrderedDict(sorted(labels.items()))

    X_test = []
    y_test = []
    for filename, clsid in ordered.items():
        x = np.load(os.path.join(avc_test_path, filename))
        X_test.append(x)
        y_test.append(clsid)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    assert X_test.dtype == np.uint8
    assert X_test.shape[1:] == (64, 64, 3)

    return X_test, y_test

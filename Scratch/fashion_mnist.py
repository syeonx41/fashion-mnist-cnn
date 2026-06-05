import gzip
import numpy as np
import os


dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels


def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)

    images = images.reshape(-1, 1, 28, 28)

    return images.astype(np.float32) / 255.0


def load_fashion_mnist():

    x_train = load_images(
        os.path.join(dataset_dir, "train-images-idx3-ubyte.gz")
    )

    t_train = load_labels(
        os.path.join(dataset_dir, "train-labels-idx1-ubyte.gz")
    )

    x_test = load_images(
        os.path.join(dataset_dir, "t10k-images-idx3-ubyte.gz")
    )

    t_test = load_labels(
        os.path.join(dataset_dir, "t10k-labels-idx1-ubyte.gz")
    )

    return (x_train, t_train), (x_test, t_test)
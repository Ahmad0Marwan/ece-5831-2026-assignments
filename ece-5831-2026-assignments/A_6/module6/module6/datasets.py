"""
datasets.py
-----------
Loaders for the Iris and MNIST datasets used in module 6.

* Iris  : loaded from dataset/iris.csv (written once from scikit-learn),
          standardised, one-hot encoded, split 80 / 20 (stratified).
* MNIST : loaded from the original IDX gzip files in dataset/,
          flattened to 784 features, scaled to [0, 1], one-hot encoded.
"""

import gzip
import os
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def one_hot(labels, num_classes):
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1
    return out


# ---------------------------------------------------------------------------
# Iris
# ---------------------------------------------------------------------------
IRIS_CSV = os.path.join(DATASET_DIR, "iris.csv")
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


def save_iris_csv():
    """Write dataset/iris.csv from scikit-learn's bundled copy (run once)."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = np.column_stack([iris.data, iris.target])
    header = ",".join(iris.feature_names + ["target"])
    np.savetxt(IRIS_CSV, data, delimiter=",", header=header, comments="", fmt="%g")
    return IRIS_CSV


def load_iris_data(test_ratio=0.2, seed=0):
    """Return (x_train, t_train), (x_test, t_test) with one-hot targets."""
    if not os.path.exists(IRIS_CSV):
        save_iris_csv()

    raw = np.loadtxt(IRIS_CSV, delimiter=",", skiprows=1)
    x, t = raw[:, :4], raw[:, 4].astype(int)

    # stratified shuffle split
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(t):
        idx = np.where(t == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    x_train, x_test = x[train_idx], x[test_idx]
    t_train, t_test = t[train_idx], t[test_idx]

    # standardise using training statistics only
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return (x_train, one_hot(t_train, 3)), (x_test, one_hot(t_test, 3))


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------
MNIST_FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}


def _load_mnist_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784)


def _load_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_mnist(normalize=True, one_hot_label=True):
    """Return (x_train, t_train), (x_test, t_test) from dataset/*.gz."""
    p = {k: os.path.join(DATASET_DIR, v) for k, v in MNIST_FILES.items()}
    x_train = _load_mnist_images(p["train_img"])
    t_train = _load_mnist_labels(p["train_label"])
    x_test = _load_mnist_images(p["test_img"])
    t_test = _load_mnist_labels(p["test_label"])

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if one_hot_label:
        t_train = one_hot(t_train, 10)
        t_test = one_hot(t_test, 10)

    return (x_train, t_train), (x_test, t_test)

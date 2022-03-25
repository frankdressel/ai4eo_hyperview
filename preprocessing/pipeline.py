import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import struct

DATA_PATH = Path("/home/hoff_nl/datasets/ai4eo/train_data/train_data")
LABEL_PATH = Path("/home/hoff_nl/datasets/ai4eo/train_data/train_gt.csv")


@dataclass
class Split:
    train_samples: np.ndarray
    test_samples: np.ndarray


def train_test_split(sample_count: int, test_split) -> Split:
    idx = np.arange(sample_count)
    np.random.seed(42)
    np.random.shuffle(idx)
    test_samples = int((sample_count * test_split))
    train_samples = sample_count - test_samples

    return Split(idx[:train_samples], idx[train_samples:])


def sample_count() -> int:
    assert len(pd.read_csv(LABEL_PATH)) == len(list(DATA_PATH.glob("*.npz")))
    return len(list(DATA_PATH.glob("*.npz")))


@dataclass
class LabelData:
    train_label_data: np.ndarray  # train samples x attributes
    test_label_data: np.ndarray  # test samples x attributes

    std_dev: np.ndarray  # attributes
    mean: np.ndarray  # attributes

    baseline_mse: np.ndarray  # just the average columnwise for score calc


def prepare_labels(label_path: Path, split: Split) -> LabelData:
    labels = pd.read_csv(label_path)
    label_arr = labels.to_numpy()[:, 1:] # remove idx column
    train_label_data = label_arr[split.train_samples, :]

    # normalization
    mean = np.mean(train_label_data)
    var = np.std(train_label_data)

    label_arr = (label_arr - mean) / var

    baseline_algo = np.mean(label_arr.mean(axis=0))
    baseline_error = ((label_arr - baseline_algo) ** 2).mean(axis=0)

    return LabelData(
        train_label_data,
        test_label_data=label_arr[split.test_samples, :],
        std_dev=var,
        mean=mean,
        baseline_mse=baseline_error
    )


@dataclass
class ImageData:
    train_images: np.ndarray  # array of prescaled images
    test_images: np.ndarray

    mean: np.ndarray  # channelwise
    std_var: np.ndarray  # channelwise


def prepare_data(data_path: Path, split: Split, preprocess_fn) -> ImageData:
    images = data_path.glob("*.npz")

    # Load all images to memory, dataset is small enough to not worry about it
    # Channels last format (transpose)
    img_data: list[np.ndarray] = [np.ma.MaskedArray(**np.load(str(image))).data.transpose((1, 2, 0)) for image in
                                  images]

    # probably possible in a more efficient way..
    train_img_data = np.array(img_data)[split.train_samples]
    test_img_data = np.array(img_data)[split.test_samples]

    means, variances = [], []
    for img in train_img_data:
        # mean axis = [0, 1] -> mean axis [0], mean axis [0]. Np doesnt do reduction along multiple axis
        means.append(img.mean(axis=0).mean(axis=0))
        variances.append(img.std(axis=0).mean(axis=0))

    mean = np.array(means).mean(axis=0)
    var = np.array(variances).mean(axis=0)

    processed_train = []
    # normalize and preprocess to single size
    for img in train_img_data:
        norm = (img - mean) / var
        preprocessed = preprocess_fn(norm)
        processed_train.append(preprocessed)

    processed_test = []
    for img in test_img_data:
        norm = (img - mean) / var
        preprocessed = preprocess_fn(norm)
        processed_test.append(preprocessed)


    return ImageData(
        np.array(processed_train),
        np.array(processed_test),
        mean,
        var
    )


@struct.dataclass
class Batch:
    image: jnp.ndarray
    label: jnp.ndarray


@dataclass
class Pipeline:
    train_generator: Callable[[], typing.Iterator[Batch]]
    test_generator: Callable[[], typing.Iterator[Batch]]

    input_shape: tuple[int]
    baseline_mse: np.ndarray  # for score calc


def get_data_pipeline(split: Split, batch_size: int, preprocess_img: Callable):
    labels = prepare_labels(LABEL_PATH, split)
    images = prepare_data(DATA_PATH, split, preprocess_img)

    train_samples_count = labels.train_label_data.shape[0]
    shuffle_idx = np.arange(train_samples_count)

    def train_generator():
        np.random.shuffle(shuffle_idx)

        # skips the last elements which dont form a complete batch. Should be fine due to shuffling
        for batch_pos in range(0, train_samples_count, batch_size):
            train_samples = shuffle_idx[batch_pos:batch_pos + batch_size]

            data_batch = images.train_images[train_samples]
            label_batch = labels.train_label_data[train_samples]

            yield Batch(data_batch, label_batch)

    def test_generator():
        for batch_pos in range(0, train_samples_count, batch_size):
            data_batch = images.train_images[batch_pos:batch_pos + batch_size]
            label_batch = labels.train_label_data[batch_pos:batch_pos + batch_size]

            yield Batch(data_batch, label_batch)

        # TODO: Just drop last incomplete batch? Or drop + shuffle? Or introduce batch masking+padding

    shape = next(train_generator()).image.shape

    return Pipeline(train_generator, test_generator, input_shape=shape, baseline_mse=labels.baseline_mse)
# At the top of the code, along with the other `import`s
from __future__ import annotations

#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

from functools import partial
from typing import List, Tuple, Callable
import tensorflow as tf


class Unpack(object):
    """ Unpack items of a dictionary to a tuple """

    def __call__(self, sample: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        return sample['features'], sample['labels']


class ToFloat32(object):
    """ Cast features to float32 """

    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        feats = tf.cast(feats, tf.float32)
        return feats, labels


class OneMinusEncoding(object):
    """ Encodes labels to 1-p, p. Makes sense only for binary labels and for continuous labels in [0, 1] """

    def __init__(self, n_classes: int):
        assert n_classes == 2, 'OneMinus works only for "binary" classes. `n_classes` should be 2.'
        self.n_classes = n_classes

    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return feats, tf.concat([tf.ones_like(labels) - labels, labels], axis=-1)


class FillNaN(object):
    """ Replace NaN values with a given finite value """

    def __init__(self, fill_value: float = -2.):
        self.fill_value = fill_value

    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        feats = tf.where(tf.math.is_nan(feats), tf.constant(
            self.fill_value, feats.dtype), feats)
        return feats, labels


class LabelsToDict(object):
    """ Convert a list of arrays to a dictionary """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[dict, dict]:
        assert len(self.keys) == labels.shape[0]
        labels_dict = {}
        for idx, key in enumerate(self.keys):
            labels_dict[key] = labels[idx, ...]
        return {'features': feats}, labels_dict


def normalize_meanstd(ds_keys: dict, subtract: str = 'mean') -> dict:
    """ 
    Help function to normalise the features by the mean and standard deviation.
    This is a normalization function that follows recent state-of-the-art solutions.
    There are two methods, that depends on mean, std = None parameters to method normalize.
    If mean, std = None, than this algorithm will be as in https://github.com/antofuller/CROMA, 
    else as in https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
    """
    # code for normalization follows parameters here
    # https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
    S2A_MEAN = {
        'B2': 889.6,
        'B3': 1151.7,
        'B4': 1307.6,
        'B8': 2538.9, }
    S2A_STD = {
        'B2': 1159.1,
        'B3': 1188.1,
        'B4': 1375.2,
        'B8': 1476.4, }
    mean = tf.constant(list(S2A_MEAN.values()), dtype=tf.float64)
    std = tf.constant(list(S2A_STD.values()), dtype=tf.float64)

    def normalize(feats, mean=None, std=None):
        # follows normalization presented https://github.com/zhu-xlab/SSL4EO-S12/tree/main
        feats = tf.cast(feats, tf.float64)
        if mean is None or std is None:
            # normalization method follows presented here https://github.com/antofuller/CROMA
            # axis to reduce, all except last where bands are
            inds = list(range(len(feats.shape) - 1))
            std = tf.math.reduce_std(feats, inds)
            mean = tf.math.reduce_mean(feats, inds)
        std_2 = tf.math.multiply(tf.constant(2.0, dtype=tf.float64), std)
        min_value = tf.math.subtract(mean, std_2)
        max_value = tf.math.add(mean, std_2)
        div = tf.math.subtract(max_value, min_value)
        feats = tf.math.subtract(feats, min_value)
        feats = tf.math.divide(feats, div)
        feats = tf.clip_by_value(feats, clip_value_min=0., clip_value_max=1.)
        return feats
    feats = normalize(ds_keys['features'], mean=mean, std=std)
    ds_keys['features'] = feats
    return ds_keys


def flip_left_right(x: tf.Tensor, flip_lr_cond: bool = False) -> tf.Tensor:
    if flip_lr_cond:
        return tf.image.flip_left_right(x)
    return x


def flip_up_down(x: tf.Tensor, flip_ud_cond: bool = False) -> tf.Tensor:
    if flip_ud_cond:
        return tf.image.flip_up_down(x)
    return x


def rotate(x: tf.Tensor, rot90_amount: int = 0) -> tf.Tensor:
    return tf.image.rot90(x, rot90_amount)


def brightness(x: tf.Tensor, brightness_delta: float = .0) -> tf.Tensor:
    return tf.image.random_brightness(x, brightness_delta)


def contrast(x: tf.Tensor, contrast_lower: float = .9, contrast_upper=1.1) -> tf.Tensor:
    return tf.image.random_contrast(x, contrast_lower, contrast_upper)


def augment_data(features_augmentations: List[str],
                 labels_augmentation: List[str],
                 brightness_delta: float = 0.1,
                 contrast_bounds: Tuple[float, float] = (0.9, 1.1)) -> Callable:
    """
    Creates a data augmentation function for features and labels.

    Args:
        features_augmentations (List[str]): List of augmentation operations to apply to features.
        labels_augmentation (List[str]): List of augmentation operations to apply to labels.
        brightness_delta (float, optional): The maximum delta for brightness adjustment. 
        Defaults to 0.1.
        contrast_bounds (Tuple[float, float], optional): The lower and upper bounds 
        for contrast adjustment. Defaults to (0.9, 1.1).

    Returns:
        Callable: A function that takes features and labels 
        as input and returns augmented features and labels.
    """
    def _augment_data(data, op_fn):
        return op_fn(data)

    def _augment_labels(labels_augmented, oper_op):
        ys = []
        for i in range(len(labels_augmented)):
            ys.append(_augment_data(labels_augmented[i, ...], oper_op))
        return tf.convert_to_tensor(ys, dtype=labels_augmented.dtype)

    def _augment(features, labels):
        contrast_lower, contrast_upper = contrast_bounds

        flip_lr_cond = tf.random.uniform(shape=[]) > 0.5
        flip_ud_cond = tf.random.uniform(shape=[]) > 0.5
        rot90_amount = tf.random.uniform(shape=[], maxval=4, dtype=tf.int32)

        # Available operations
        operations = {
            'flip_left_right': partial(flip_left_right, flip_lr_cond=flip_lr_cond),
            'flip_up_down': partial(flip_up_down, flip_ud_cond=flip_ud_cond),
            'rotate': partial(rotate, rot90_amount=rot90_amount),
            'brightness': partial(brightness, brightness_delta=brightness_delta),
            'contrast': partial(contrast, contrast_lower=contrast_lower, contrast_upper=contrast_upper)
        }

        for op in features_augmentations:
            features = _augment_data(features, operations[op])

        for op in labels_augmentation:
            labels = _augment_labels(labels, operations[op])

        return features, labels

    return _augment

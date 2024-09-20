import os
import logging
from pathlib import Path
import tensorflow as tf
from tqdm.auto import tqdm
from functools import partial
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tf_data_utils import normalize_meanstd, augment_data
from tf_data_utils import Unpack, ToFloat32, FillNaN, OneMinusEncoding, LabelsToDict
import numpy as np
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Paths parameters
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
NPZ_FILES_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/npz_files/')
DATASET_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/datasets/')
METADATA_PATH = Path(f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe.csv')

# Dataset generation parameters
NORMALIZER = dict(to_medianstd=partial(normalize_meanstd, subtract='median'))
ENABLE_AUGMENTATION = True
AUGMENTATIONS_FEATURES = ["flip_left_right",
                          "flip_up_down", "rotate", "brightness"]
AUGMENTATIONS_LABEL = ["flip_left_right", "flip_up_down", "rotate"]

N_FOLDS = int(os.getenv('N_FOLDS', 10))
PROCESS_POOL_WORKERS = int(os.getenv('PROCESS_POOL_WORKERS', os.cpu_count()))
USE_FILE_SHARDING = False
NUM_SHARDS = int(os.getenv('NUM_SHARDS', 35))
SHUFFLE_BUFFER_SIZE = 2000
INTERLEAVE_CYCLE_LENGTH = 10


def describe_tf_dataset(dataset, dataset_name, message, num_batches=3):
    """
    Describes a TensorFlow dataset.

    Args:
        dataset: The TensorFlow dataset to describe.
        dataset_name: The name of the dataset.
        message: A message to include in the description.
        num_batches: The number of batches to inspect (default is 3).

    Returns:
        None
    """
    LOGGER.debug(f"Inspection of dataset {dataset_name} : {message}")
    LOGGER.debug("Dataset description:")
    LOGGER.debug(f"Type: {type(dataset)}")
    try:
        length = sum(1 for _ in dataset)
        LOGGER.debug(f"Number of elements (length): {length}")
    except:
        LOGGER.debug(
            "Number of elements (length): Unable to determine (infinite or dynamically generated dataset)")
    LOGGER.debug(f"Taking a look at the first {num_batches} batches:")
    for i, batch in enumerate(dataset.take(num_batches)):
        LOGGER.debug(f"\nBatch {i+1} summary:")
        inspect_structure(batch)
        if i+1 >= num_batches:
            break


def inspect_structure(element, prefix=''):
    """
    Inspects the structure of the given element.

    Parameters:
        element (tf.Tensor, dict, tuple, list): The element to inspect.
        prefix (str): The prefix to add to the log messages.

    Returns:
        None

    Raises:
        None
    """
    if isinstance(element, tf.Tensor):
        LOGGER.debug(f"{prefix}Shape: {element.shape}, Dtype: {element.dtype}")
        LOGGER.debug(f"{prefix}First element data: {element.numpy()[0]}")
    elif isinstance(element, dict):
        for key, value in element.items():
            LOGGER.debug(f"{prefix}{key}:")
            inspect_structure(value, prefix='  ' + prefix)
    elif isinstance(element, (tuple, list)):
        for i, value in enumerate(element):
            LOGGER.debug(f"{prefix}Element {i}:")
            inspect_structure(value, prefix='  ' + prefix)
    else:
        LOGGER.debug(f"{prefix}Unsupported element type: {type(element)}")


def npz_file_lazy_dataset(file_path: str,
                          fields: List[str],
                          types: List[tf.dtypes.DType],
                          shapes: List[np.int32]) -> tf.data.Dataset:
    """
    Creates a TensorFlow tf.data.Dataset from a NumPy .npz file.

    Args:
        file_path (str): The path to the .npz file.
        fields (List[str]): The names of the fields to load from the .npz file.
        types (List[tf.dtypes.DType]): The data types of the fields.
        shapes (List[np.int32]): The shapes of the fields.

    Returns:
        tf.data.Dataset: The TensorFlow dataset containing the loaded data.

    Raises:
        AssertionError: If the arrays in the .npz file do not have matching first dimensions.
    """
    def _generator():
        data = np.load(file_path)
        np_arrays = [data[f] for f in fields]
        # Check that arrays match in the first dimension
        n_samples = np_arrays[0].shape[0]
        assert all(n_samples == arr.shape[0] for arr in np_arrays)
        # Yield each sample (slice) as a tuple from the arrays
        for slices in zip(*np_arrays):
            yield slices

    output_signature = tuple(
        tf.TensorSpec(shape=shape, dtype=dtype) for shape, dtype in zip(shapes, types)
    )
    ds = tf.data.Dataset.from_generator(
        _generator, output_signature=output_signature)

    # Converts a tuple of features into a dict with 'features' and 'labels'
    def _to_dict(features, y_extent, y_boundary, y_distance):
        return {
            'features': features,
            'labels': [y_extent, y_boundary, y_distance]
        }

    # Apply the map function to convert the tuples to the dict structure
    ds = ds.map(_to_dict)
    return ds


def get_dataset(npz_folder: str, fold_type: str) -> tf.data.Dataset:
    """
    Retrieves a TensorFlow dataset from a folder containing npz files.

    Args:
        npz_folder (str): The path to the folder containing npz files.
        fold_type (str): The type of fold.

    Returns:
        tf.data.Dataset: The TensorFlow dataset.

    Raises:
        ValueError: If no npz files are found in the given folder.
    """
    files = [os.path.join(npz_folder, f)
             for f in os.listdir(npz_folder) if f.endswith('.npz')]
    if not files:
        raise ValueError(f"No npz files found in {npz_folder}")

    fields = ["features", "y_extent", "y_boundary", "y_distance"]

    # Read one file for shape and type info
    file_test = files[0]
    data_test = np.load(file_test)
    np_arrays = [data_test[f] for f in fields]
    shapes = tuple(arr.shape[1:] for arr in np_arrays)
    types = (tf.uint16, tf.float32, tf.float32, tf.float32)

    # Create datasets from npz files
    datasets = [npz_file_lazy_dataset(
        npz_file, fields, types, shapes) for npz_file in files]
    dataset = tf.data.Dataset.from_tensor_slices(datasets)

    # Shuffle files and interleave multiple files in parallel
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    dataset = dataset.interleave(lambda x: x,
                                 cycle_length=INTERLEAVE_CYCLE_LENGTH,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if LOGGER.level == logging.DEBUG:
        describe_tf_dataset(dataset, fold_type, "Before augmentation")
    normalizer = NORMALIZER["to_medianstd"]

    augmentations = [augment_data(
        AUGMENTATIONS_FEATURES, AUGMENTATIONS_LABEL)] if ENABLE_AUGMENTATION else []
    dataset_ops = (
        [normalizer, Unpack(), ToFloat32()] +
        augmentations +
        [FillNaN(fill_value=-2),
         OneMinusEncoding(n_classes=2),
         LabelsToDict(["extent", "boundary", "distance"])]
    )

    for dataset_op in dataset_ops:
        dataset = dataset.map(
            dataset_op, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if LOGGER.level == logging.DEBUG:
            op_str = dataset_op.__name__ if hasattr(
                dataset_op, '__name__') else str(dataset_op)
            describe_tf_dataset(dataset, fold_type, f"After {op_str}")

    if LOGGER.level == logging.DEBUG:
        describe_tf_dataset(dataset, fold_type, "After all operations")
    return dataset


def shard_func(element, index):
    return tf.random.uniform((), minval=0, maxval=NUM_SHARDS, dtype=tf.int64)


def create_datasets():
    for fold_type in ["train", "val", "test"]:
        npz_folder = NPZ_FILES_DIR / fold_type
        dataset_folder = DATASET_DIR / fold_type
        if dataset_folder.exists():
            LOGGER.info(
                f"Dataset folder {dataset_folder} already exists, cleaning up")
            shutil.rmtree(dataset_folder)
        dataset_folder.mkdir(parents=True, exist_ok=True)
        dataset_folder_path = dataset_folder.as_posix()
        dataset = get_dataset(npz_folder, fold_type)
        if USE_FILE_SHARDING:
            dataset.save(path=dataset_folder_path, shard_func=shard_func)
        else:
            dataset.save(path=dataset_folder_path)
        LOGGER.info(
            f"Saved dataset for fold {fold_type} to {dataset_folder_path}")


if __name__ == '__main__':
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable is not set")
    create_datasets()

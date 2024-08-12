import os
import logging
from pathlib import Path
import tensorflow as tf
from tqdm.auto import tqdm
from functools import partial
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from tf_data_utils import (
    npz_dir_dataset,
    normalize_meanstd,
    Unpack, ToFloat32, augment_data, FillNaN, OneMinusEncoding, LabelsToDict)

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Paths parameters
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
FOLDS_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/folds/')
DATASET_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/datasets/')
METADATA_PATH = Path(f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe_final.csv')

# Dataset generation parameters
NORMALIZER = dict(to_medianstd=partial(normalize_meanstd, subtract='median'))
AUTOTUNE = tf.data.experimental.AUTOTUNE
AUGMENTATIONS_FEATURES = ["flip_left_right",
                          "flip_up_down", "rotate", "brightness"]
AUGMENTATIONS_LABEL = ["flip_left_right", "flip_up_down", "rotate"]
N_FOLDS = int(os.getenv('N_FOLDS', 10))
PROCESS_POOL_WORKERS = int(os.getenv('PROCESS_POOL_WORKERS', os.cpu_count()))
NUM_SHARDS = int(os.getenv('NUM_SHARDS', 35))
USE_FILE_SHARDING = False


def describe_tf_dataset(dataset, dataset_name, message, num_batches=5):
    """
    Print and log a summary of a TensorFlow dataset.

    :param dataset: tf.data.Dataset to describe.
    :param num_batches: Number of batches to inspect for summary statistics.
    :param dataset_name: Name of the dataset to use in the log file name.
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
    Recursively inspect the structure of dataset elements and log the information.
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


def get_dataset(fold_folder, metadata_path, fold, augment,
                augmentations_features, augmentations_label,
                num_parallel, randomize=True):
    """
    Creates and returns an augmented dataset for the given fold.

    Args:
        fold_folder (str): Path to the folder containing folds data.
        metadata_path (str): Path to the metadata file.
        fold (int): Fold number.
        augment (bool): Whether to apply data augmentation.
        augmentations_features (list): List of feature augmentations to apply.
        augmentations_label (list): List of label augmentations to apply.
        num_parallel (int): Number of parallel processes to use for dataset interleave.
        randomize (bool, optional): Whether to randomize the dataset. Defaults to True.

    Returns:
        tf.data.Dataset: The created dataset.

    """
    data = dict(X='features', y_extent='y_extent',
                y_boundary='y_boundary', y_distance='y_distance')
    fold_folder_path = os.path.join(fold_folder, f'fold_{fold}')
    dataset = npz_dir_dataset(file_dir_or_list=fold_folder_path,
                              features=data,
                              metadata_path=metadata_path,
                              fold=fold,
                              randomize=randomize,
                              num_parallel=num_parallel)
    if LOGGER.level == logging.DEBUG:
        describe_tf_dataset(dataset, f"fold_{fold}", "Before augmentation")
    normalizer = NORMALIZER["to_medianstd"]

    augmentations = [augment_data(
        augmentations_features, augmentations_label)] if augment else []
    dataset_ops = [normalizer, Unpack(), ToFloat32()] \
        + augmentations \
        + [FillNaN(fill_value=-2),
           OneMinusEncoding(
            n_classes=2),
           LabelsToDict(["extent", "boundary", "distance"])]

    for dataset_op in dataset_ops:
        dataset = dataset.map(dataset_op, num_parallel_calls=AUTOTUNE)
        if LOGGER.level == logging.DEBUG:
            op_str = dataset_op.__name__ if hasattr(
                dataset_op, '__name__') else str(dataset_op)
            describe_tf_dataset(dataset, f"fold_{fold}", f"After {op_str}")

    if LOGGER.level == logging.DEBUG:
        describe_tf_dataset(dataset, f"fold_{fold}", "After all operations")
    return dataset


@tf.function
def shard_func(element, index):
    """Custom shard function that assigns shard IDs to dataset elements using a random distribution."""
    return tf.random.uniform((), minval=0, maxval=NUM_SHARDS, dtype=tf.int64)


def save_dataset(folds_folder, metadata_path, dataset_folder, fold,
                 augmentations_features, augmentations_label, num_parallel):
    """
    Saves the dataset for the given fold to the specified folder.

    Args:
        folds_folder (str): Path to the folder containing folds.
        metadata_path (str): Path to the metadata file.
        dataset_folder (str): Path to the folder where the dataset will be saved.
        fold (int): Fold number, used to identify the fold data.
        augmentations_features (list): List of feature augmentations to apply.
        augmentations_label (list): List of label augmentations to apply.
        num_parallel (int): Number of parallel processes to use for dataset interleave.

    Returns:
        None
    """
    dataset = get_dataset(folds_folder, metadata_path, fold, True,
                          augmentations_features, augmentations_label, num_parallel)

    dataset_path = os.path.join(dataset_folder, f'fold_{fold}')
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    try:
        if USE_FILE_SHARDING:
            dataset.save(path=dataset_path, shard_func=shard_func)
        else:
            dataset.save(path=dataset_path)
        LOGGER.info(f'Saved dataset for fold {fold} to {dataset_path}')
    except Exception as e:
        LOGGER.error(
            f'Error saving dataset for fold {fold} to {dataset_path}: {e}')
        raise


def save_datasets_parallel(folds_folder, metadata_path, dataset_folder,
                           augmentations_features, augmentations_label,
                           n_folds, num_parallel=AUTOTUNE):
    """
    Saves the datasets for all folds in parallel.

    Args:
        folds_folder (str): Path to the folder containing folds.
        metadata_path (str): Path to the metadata file.
        dataset_folder (str): Path to the folder where datasets will be saved.
        augmentations_features (list): List of feature augmentations to apply.
        augmentations_label (list): List of label augmentations to apply.
        n_folds (int): Number of folds.
        num_parallel (int, optional): Number of parallel processes to use for dataset interleave. 
        Defaults to AUTOTUNE.

    Returns:
        None
    """
    os.makedirs(dataset_folder, exist_ok=True)
    futures = []
    with ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS) as executor:
        for fold in range(1, n_folds + 1):
            futures.append(executor.submit(save_dataset, folds_folder, metadata_path,
                                           dataset_folder, fold, augmentations_features,
                                           augmentations_label, num_parallel))
        for future in tqdm(as_completed(futures), total=n_folds):
            try:
                future.result()
            except Exception as e:
                LOGGER.error(f"Error occurred: {e}")
                exit(1)


def create_datasets():
    if LOGGER.level == logging.DEBUG:
        # sequential for debugging
        LOGGER.debug("Warning: Debug mode enabled, running sequentially")
        for fold in range(1, N_FOLDS + 1):
            save_dataset(FOLDS_FOLDER, METADATA_PATH, DATASET_FOLDER, fold,
                         AUGMENTATIONS_FEATURES, AUGMENTATIONS_LABEL, AUTOTUNE)
    else:
        # parallel otherwise
        save_datasets_parallel(FOLDS_FOLDER, METADATA_PATH, DATASET_FOLDER,
                               AUGMENTATIONS_FEATURES, AUGMENTATIONS_LABEL, N_FOLDS)


if __name__ == '__main__':
    create_datasets()

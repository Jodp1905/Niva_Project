import random
import os
import logging
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List
from pathlib import Path
from eolearn.core import EOPatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)

# Load configuration
from niva_utils.config_loader import load_config  # noqa: E402
CONFIG = load_config()

# Constants
NIVA_PROJECT_DATA_ROOT = CONFIG['niva_project_data_root']
NPZ_CHUNK_SIZE = CONFIG['create_npz']['npz_chunk_size']

# Inferred constants
EOPTACHES_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/training_data/eopatches/')
NPZ_FILES_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/training_data/npz_files/')
METADATA_PATH = Path(
    f'{NIVA_PROJECT_DATA_ROOT}/training_data/patchlets_dataframe.csv')
PROCESS_POOL_WORKERS = os.cpu_count()


def extract_npys(patchlet_path: str) -> Tuple:
    """
    Extracts numpy arrays and other data from the given patchlet path.

    Args:
        patchlet_path (str): The path to the patchlet directory.

    Returns:
        Tuple: A tuple containing the following elements:
            - X_data (numpy.ndarray): The data array loaded from 'BANDS.npy'.
            - y_boundary (numpy.ndarray): The boundary mask array loaded from 'BOUNDARY.npy'.
            - y_extent (numpy.ndarray): The extent mask array loaded from 'EXTENT.npy'.
            - y_distance (numpy.ndarray): The distance mask array loaded from 'DISTANCE.npy'.
            - timestamps (list): The timestamps extracted from the EOPatch.
            - eop_names (numpy.ndarray): An array containing the patchlet path
            repeated for each timestamp.
    """
    try:
        X_data = np.load(os.path.join(patchlet_path, 'data', 'BANDS.npy'))
        y_boundary = np.load(os.path.join(
            patchlet_path, 'data_timeless', 'BOUNDARY.npy'))
        y_extent = np.load(os.path.join(
            patchlet_path, 'data_timeless', 'EXTENT.npy'))
        y_distance = np.load(os.path.join(
            patchlet_path, 'data_timeless', 'DISTANCE.npy'))

        # Repeat the timeless masks along the time dimension
        eop = EOPatch.load(patchlet_path, lazy_loading=True)
        timestamps = eop.timestamp
        time_steps = len(timestamps)
        y_boundary = np.repeat(y_boundary[np.newaxis, ...], time_steps, axis=0)
        y_extent = np.repeat(y_extent[np.newaxis, ...], time_steps, axis=0)
        y_distance = np.repeat(y_distance[np.newaxis, ...], time_steps, axis=0)
        eop_names = np.repeat([patchlet_path], len(timestamps), axis=0)

    except Exception as e:
        LOGGER.error(f"Could not create for {patchlet_path}. Exception {e}")
        return None, None, None, None, None, None

    return X_data, y_boundary, y_extent, y_distance, timestamps, eop_names


def process_chunk(chunk_patchlets: List[str], chunk_index: int, output_folder: str) -> None:
    """
    Process a chunk of patchlets and save the results as npz files.

    Args:
        chunk_patchlets (List[str]): List of paths to the patchlet files.
        chunk_index (int): Index of the chunk being processed.
        output_folder (str): Path to the folder where the npz files will be saved.

    Returns:
        df (pd.DataFrame): A DataFrame containing the metadata of the saved chunk.
    """
    results = []
    for patchlet_path in chunk_patchlets:
        result = extract_npys(patchlet_path)
        if any(x is None for x in result):
            LOGGER.warning(f"Failed to extract data from {patchlet_path}")
        else:
            results.append(result)
    if not results:
        LOGGER.warning(f"No valid results in chunk {chunk_index}.")
        return
    npys_dict = concatenate_npys(results)
    df: pd.DataFrame = save_chunk(npys_dict, chunk_index, output_folder)
    return df


def concatenate_npys(results: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenates the arrays from the given list of tuples.

    Args:
        results (List[Tuple]): A list of tuples containing arrays to be concatenated.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple of concatenated arrays.

    Raises:
        None

    """
    if not results:
        return None, None, None, None, None, None
    X, y_boundary, y_extent, y_distance, timestamps, eop_names = zip(*results)
    X = np.concatenate(X)
    y_boundary = np.concatenate(y_boundary)
    y_extent = np.concatenate(y_extent)
    y_distance = np.concatenate(y_distance)
    timestamps = np.concatenate(timestamps)
    eop_names = np.concatenate(eop_names)
    return X, y_boundary, y_extent, y_distance, timestamps, eop_names


def save_chunk(npys_dict: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray],
               chunk_index: int,
               output_folder: str) -> None:
    """
    Save the chunk data as numpy arrays and update the metadata file.
    Timestamps and eopatches paths are not saved in the numpy arrays.
    They are saved in the metadata dataframe.

    Args:
        npys_dict (Tuple[np.ndarray, np.ndarray, np.ndarray, 
                         np.ndarray, np.ndarray, np.ndarray]):
            A tuple containing the numpy arrays to be saved.
            The elements of the tuple are:
                - X: Input data array
                - y_boundary: Boundary data array
                - y_extent: Extent data array
                - y_distance: Distance data array
                - timestamps: Timestamps data array
                - eopatches: Eopatches paths
        chunk_index (int): The index of the chunk.
        output_folder (str): The path to the output folder.

    Returns:
        df (pd.DataFrame): A DataFrame containing the metadata of the saved chunk.
    """
    eopatches = [os.path.basename(eop) for eop in npys_dict[5]]
    filename = f'eopatch_chunk_{chunk_index}'
    timestamps = pd.to_datetime(npys_dict[4], utc=True).tz_localize(None)
    np.savez(os.path.join(output_folder, f'{filename}.npz'),
             features=npys_dict[0],
             y_boundary=npys_dict[1],
             y_extent=npys_dict[2],
             y_distance=npys_dict[3])
    df = pd.DataFrame(
        dict(chunk=[f'{filename}.npz'] * len(npys_dict[5]),
             eopatch=eopatches,
             patchlet=npys_dict[5],
             chunk_pos=[i for i in range(len(npys_dict[5]))],
             timestamp=timestamps))
    return df


def eopatches_to_npz_files():
    """
    Processes EO patches and saves them as NPZ files.

    This function performs the following steps:
    1. Creates the NPZ files directory if it doesn't exist.
    2. Iterates over the "train", "val", and "test" folds.
    3. Cleans the output directory for each fold.
    4. Computes chunks of EO patches based on a predefined chunk size.
    5. Processes and saves each chunk in parallel.
    6. Create a master DataFrame containing metadata on the origin of each npz file entry.

    Raises:
        ValueError: If the number of entries does not match the expected number of EO patches.
    """
    NPZ_FILES_DIR.mkdir(parents=True, exist_ok=True)
    df_master_list = []
    for fold in ["train", "val", "test"]:
        eopatch_dir = EOPTACHES_DIR / fold
        eopatches_paths = [
            eopatch for eopatch in eopatch_dir.iterdir() if eopatch.is_dir()]
        random.shuffle(eopatches_paths)

        # Clean output directory
        npz_dir = NPZ_FILES_DIR / fold
        if npz_dir.exists():
            LOGGER.info(
                f'Detecting existing npz files directory: {npz_dir}, cleaning up')
            shutil.rmtree(npz_dir)
        npz_dir.mkdir(parents=True, exist_ok=True)

        # Compute chunks
        nb_chunks = len(eopatches_paths) // NPZ_CHUNK_SIZE
        lost = len(eopatches_paths) % NPZ_CHUNK_SIZE
        chunks = [eopatches_paths[i:i + NPZ_CHUNK_SIZE]
                  for i in range(0, len(eopatches_paths) - lost, NPZ_CHUNK_SIZE)]

        LOGGER.info(
            f'Processing {len(eopatches_paths)} patchlets in {nb_chunks} chunks '
            f'containing {NPZ_CHUNK_SIZE} eopatches each, with {lost} eopatches unused.')

        # Process chunks in parallel
        df_list = []
        with ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS) as executor:
            futures = []
            for chunk_index, chunk in enumerate(chunks):
                future = executor.submit(
                    process_chunk, chunk, chunk_index, npz_dir)
                futures.append(future)
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Processing chunks"):
                try:
                    df = future.result()
                    df_list.append(df)
                except Exception as e:
                    LOGGER.error(f'A task failed: {e}')
        if df_list:
            df_concatenated = pd.concat(df_list).reset_index(drop=True)
            df_concatenated['fold'] = fold
            df_master_list.append(df_concatenated)
        LOGGER.info(
            f'Finished processing {fold} fold, saved npz files to {npz_dir}')
    master_df = pd.concat(df_master_list).reset_index(drop=True)
    num_eopatches = len(master_df['patchlet'].unique())
    num_entries = len(master_df)
    if num_entries != num_eopatches * 6:
        raise ValueError(
            f'Number of entries {num_entries} does not match the expected number '
            f'of eopatches {num_eopatches} * 6 = {num_eopatches * 6}')
    LOGGER.info(
        f'Finished processing all {num_eopatches} eopatches, flattened by time '
        f'to {num_entries} entries')
    master_df.to_csv(METADATA_PATH, index=False)
    LOGGER.info(f'saved metadata to {METADATA_PATH}')


if __name__ == '__main__':
    if NIVA_PROJECT_DATA_ROOT is None:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable should be set.")
    eopatches_to_npz_files()

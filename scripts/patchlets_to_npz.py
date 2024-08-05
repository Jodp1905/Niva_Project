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
import random

from filter import LogFileFilter

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Define paths
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
PATCHLETS_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/eopatches/')
NPZ_FILES_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/npz_files/')
METADATA_PATH = Path(f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe.csv')

# Parameters
# Number of patchlets data concatenated in each .npz end file
NPZ_NB_CHUNKS = int(os.getenv('NPZ_NB_CHUNKS', 100))
PROCESS_POOL_WORKERS = int(os.getenv('PROCESS_POOL_WORKERS', os.cpu_count()))


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
            patchlet_path, 'mask_timeless', 'BOUNDARY.npy'))
        y_extent = np.load(os.path.join(
            patchlet_path, 'mask_timeless', 'EXTENT.npy'))
        y_distance = np.load(os.path.join(
            patchlet_path, 'mask_timeless', 'DISTANCE.npy'))

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

    Args:
        npys_dict (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
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
    filename = f'patchlets_fd_{chunk_index}'
    timestamps = pd.to_datetime(npys_dict[4], utc=True).tz_localize(None)
    np.savez(os.path.join(output_folder, f'{filename}.npz'),
             X=npys_dict[0],
             y_boundary=npys_dict[1],
             y_extent=npys_dict[2],
             y_distance=npys_dict[3],
             timestamps=timestamps,
             eopatches=npys_dict[5])
    df = pd.DataFrame(
        dict(chunk=[f'{filename}.npz'] * len(npys_dict[5]),
             eopatch=eopatches,
             patchlet=npys_dict[5],
             chunk_pos=[i for i in range(len(npys_dict[5]))],
             timestamp=timestamps))
    return df


def patchlets_to_npz_files():
    """
    Convert patchlets to NPZ files.

    This function converts a list of patchlets to NPZ files. It performs the following steps:
    1. Retrieves the paths of all patchlets in the PATCHLETS_DIR directory.
    2. Cleans the output directory.
    3. Divides the patchlet paths into chunks of size NPZ_CHUNK_SIZE.
    4. Processes each chunk in parallel using a ProcessPoolExecutor.
    5. Concatenates the resulting dataframes and saves them as a CSV file.

    Parameters:
    None

    Returns:
    None
    """
    patchlet_paths = [
        patchlet for patchlet in PATCHLETS_DIR.iterdir() if patchlet.is_dir()]
    random.shuffle(patchlet_paths)

    # Clean output directory
    LOGGER.info(f'Cleaning output directory {NPZ_FILES_DIR}')
    NPZ_FILES_DIR.mkdir(parents=True, exist_ok=True)
    for item in NPZ_FILES_DIR.iterdir():
        if item.is_dir() and item.name.startswith('eopatch_'):
            shutil.rmtree(item)

    # Compute chunks
    chunk_size = len(patchlet_paths) // NPZ_NB_CHUNKS
    lost = len(patchlet_paths) % NPZ_NB_CHUNKS
    chunks = [patchlet_paths[i:i + chunk_size]
              for i in range(0, len(patchlet_paths) - lost, chunk_size)]

    LOGGER.info(
        f'Processing {len(patchlet_paths)} patchlets in {len(chunks)} chunks '
        f'of size {chunk_size} each, with {lost} patchlets lost.')

    # Process chunks in parallel
    df_list = []
    with ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS) as executor:
        futures = []
        for chunk_index, chunk in enumerate(chunks):
            future = executor.submit(
                process_chunk, chunk, chunk_index, NPZ_FILES_DIR)
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            try:
                df = future.result()
                df_list.append(df)
            except Exception as e:
                LOGGER.error(f'A task failed: {e}')
    LOGGER.info('All chunks processed, writing metadata...')
    if df_list:
        df_concatenated = pd.concat(df_list).reset_index(drop=True)
        df_concatenated.to_csv(METADATA_PATH, index=True,
                               index_label='index', header=True)
    LOGGER.info(f'Saved metadata to {METADATA_PATH}')


if __name__ == '__main__':
    if NIVA_PROJECT_DATA_ROOT is None:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable should be set.")
    patchlets_to_npz_files()

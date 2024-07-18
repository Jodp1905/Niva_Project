import sys
import os
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import numpy as np
from numpy.random import default_rng
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Define paths
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
NPZ_FILES_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/npz_files/')
NORMALIZED_METADATA_PATH = Path(
    f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe_normalized.csv')
FINAL_METADATA_PATH = Path(
    f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe_final.csv')
KFOLD_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/folds/')

# Parameters
NUM_FOLDS = os.getenv('NUM_FOLDS', 10)


def fold_split(chunk: str, df: pd.DataFrame, npz_folder: str, folds_folder: str, n_folds: int):
    """
    Reads npz data from a chunk and splits it into n_folds.

    Args:
        chunk (str): The name of the chunk.
        df (pd.DataFrame): The DataFrame containing the data.
        npz_folder (str): The folder path where the npz files are stored.
        folds_folder (str): The folder path where the folds will be saved.
        n_folds (int): The number of folds to create.

    Returns:
        None
    """
    chunk = chunk + '.npz'
    data = np.load(os.path.join(npz_folder, chunk), allow_pickle=True)
    for fold in range(1, n_folds + 1):
        idx_fold = df[(df.chunk == chunk) & (df.fold == fold)].chunk_pos
        if not idx_fold.empty:
            patchlets = {key: data[key][idx_fold] for key in data}
            fold_folder = os.path.join(folds_folder, f'fold_{fold}')
            fold_chunk_name = f'{chunk}_fold_{fold}.npz'
            np.savez(os.path.join(fold_folder, fold_chunk_name), **patchlets)


def k_folds(metadata_path: str, npz_folder: str,
            n_folds: int, folds_folder: Path):
    """
    Splits the data into k-folds and assigns each data point to a specific fold.

    Args:
        metadata_path (str): The path to the metadata file.
        npz_folder (str): The path to the folder containing the npz files.
        n_folds (int): The number of folds to create.
        folds_folder (Path): The path to the folder where the fold folders will be created.

    Returns:
        None
    """
    LOGGER.info(f'Read metadata file {metadata_path}')
    df = pd.read_csv(metadata_path)
    eops = df.eopatch.unique()

    LOGGER.info('Assign folds to eopatches')
    # Randomly assign folds to patchlets
    rng = default_rng()
    fold = rng.integers(1, n_folds + 1, size=len(eops))
    eopatch_to_fold_map = dict(zip(eops, fold))
    df['fold'] = df['eopatch'].apply(lambda x: eopatch_to_fold_map[x])
    for nf in range(n_folds):
        LOGGER.info(f'{len(df[df.fold == nf + 1])} patchlets in fold {nf + 1}')

    # Create fold folders
    for fold in range(1, n_folds + 1):
        fold_folder: Path = folds_folder / f'fold_{fold}'
        fold_folder.mkdir(parents=True, exist_ok=True)

    partial_fn = partial(fold_split, df=df, npz_folder=npz_folder,
                         folds_folder=folds_folder, n_folds=n_folds)
    npz_files = sorted([file.stem for file in NPZ_FILES_DIR.glob('*.npz')])

    LOGGER.info('Splitting patchlets into folds')
    with ProcessPoolExecutor() as executor:
        futures = []
        with tqdm(total=len(npz_files)) as pbar:
            for npz_file in npz_files:
                future = executor.submit(partial_fn, npz_file)
                futures.append(future)
                pbar.update(1)
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f'A task failed: {e}')

    LOGGER.info('Saving metadata file')
    df.to_csv(FINAL_METADATA_PATH, index=False)


if __name__ == '__main__':
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError('NIVA_PROJECT_DATA_ROOT environment variable not set')
    k_folds(NORMALIZED_METADATA_PATH, NPZ_FILES_DIR,
            NUM_FOLDS, KFOLD_FOLDER)

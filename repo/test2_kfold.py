import os
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def split_into_k_folds(metadata_path: str, n_folds: int, seed: int) -> pd.DataFrame:
    """ Loads the dataframe with patchlets descriptions and splits into k-folds """
    LOGGER.info(f"Loading patchlets metadata from {metadata_path} and splitting into {n_folds} folds with seed {seed}")

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Apply k-fold split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_column = f'fold_{n_folds}'

    df[fold_column] = -1  # Initialize fold column with -1

    for fold_index, (_, test_index) in enumerate(kf.split(df)):
        df.loc[test_index, fold_column] = fold_index + 1  # Assign fold numbers (1 to n_folds)

    return df


def update_metadata_with_folds(metadata_path: str, df: pd.DataFrame):
    """ Updates the info csv file with fold information """
    LOGGER.info(f"Updating metadata file {metadata_path} with fold information")
    df.to_csv(metadata_path, index=False)


def k_folds(metadata_path: str, n_folds: int, seed: int):
    """ Function to load metadata, split into k-folds, update metadata file, and display fold paths """
    LOGGER.info(f"Starting process with metadata {metadata_path}, {n_folds} folds, seed {seed}")

    # Split into k-folds
    df = split_into_k_folds(metadata_path, n_folds, seed)

    # Update metadata file with fold information
    update_metadata_with_folds(metadata_path, df)

    # Display fold paths
    NPZ_FILES_FOLDER = '/path/to/npz_files'  # Specify your npz_files folder path here

    for fold in range(n_folds):
        fold_path = os.listdir(os.path.join(NPZ_FILES_FOLDER, f'fold_{fold+1}'))
        LOGGER.info(f'In Fold {fold+1}:')
        LOGGER.info(fold_path)

    LOGGER.info("Process completed successfully.")


if __name__ == '__main__':
    # Définition paramètres
    metadata_path = '/home/joseph/Code/localstorage/dataframe.csv'
    n_folds = 3
    seed = 2

    # Appel à la fonction k_folds
    k_folds(metadata_path, n_folds, seed)

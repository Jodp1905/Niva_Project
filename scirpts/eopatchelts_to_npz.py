import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

from eolearn.core import EOPatch

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def extract_npys_from_folder(patchlets_folder: str) -> List[Tuple]:
    """ Extract X, y_boundary, y_extent, y_distance, timestamps, eop_names from all patchlets in patchlets_folder. """
    results = []
    for patchlet_folder in os.listdir(patchlets_folder):
        patchlet_path = os.path.join(patchlets_folder, patchlet_folder)
        if os.path.isdir(patchlet_path):
            result = extract_npys(patchlet_path)
            if any(x is None for x in result):  # Check if any element in result tuple is None
                LOGGER.warning(f"Failed to extract data from {patchlet_folder}")
                print(f"Failed to extract data from {patchlet_folder}")
            else:
                results.append(result)

    return results


def extract_npys(patchlet_path: str) -> Tuple:
    """ Return X, y_boundary, y_extent, y_distance, timestamps, eop_names numpy arrays for this patchlet."""
    try:
        X_data = np.load(os.path.join(patchlet_path, 'data', 'BANDS.npy'))
        y_boundary = np.load(os.path.join(patchlet_path, 'mask_timeless', 'BOUNDARY.npy'))
        y_extent = np.load(os.path.join(patchlet_path, 'mask_timeless', 'EXTENT.npy'))
        y_distance = np.load(os.path.join(patchlet_path, 'mask_timeless', 'DISTANCE.npy'))

        eop = EOPatch.load(patchlet_path, lazy_loading=True)
        timestamps = eop.timestamp
        eop_names = np.repeat([patchlet_path], len(timestamps), axis=0)

    except Exception as e:
        LOGGER.error(f"Could not create for {patchlet_path}. Exception {e}")
        return None, None, None, None, None, None

    return X_data, y_boundary, y_extent, y_distance, timestamps, eop_names


def concatenate_npys(results: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Concatenate numpy arrays from each patchlet into one big numpy array"""
    if not results:
        LOGGER.warning("No valid results to concatenate.")
        print("No valid results to concatenate.")
        return None, None, None, None, None, None
    
    X, y_boundary, y_extent, y_distance, timestamps, eop_names = zip(*results)

    X = np.concatenate(X)
    y_boundary = np.concatenate(y_boundary)
    y_extent = np.concatenate(y_extent)
    y_distance = np.concatenate(y_distance)
    timestamps = np.concatenate(timestamps)
    eop_names = np.concatenate(eop_names)

    return X, y_boundary, y_extent, y_distance, timestamps, eop_names


def save_into_chunks(output_folder: str, npys_dict: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], chunk_size: int, output_dataframe: str) -> None:
    eopatches = [os.path.basename("_".join(x.split("_")[:-1])) for x in npys_dict[5]]
    chunk_counter = 0
    dfs = []

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    for i in range(0, len(npys_dict[0]), chunk_size):
        filename = f'patchlets_field_delineation_{chunk_counter}'
        np.savez(os.path.join(output_folder, f'{filename}.npz'),
                    X=npys_dict[0][i:i + chunk_size],
                    y_boundary=npys_dict[1][i:i + chunk_size],
                    y_extent=npys_dict[2][i:i + chunk_size],
                    y_distance=npys_dict[3][i:i + chunk_size],
                    timestamps=npys_dict[4][i:i + chunk_size],
                    eopatches=npys_dict[5][i:i + chunk_size])

        dfs.append(pd.DataFrame(dict(chunk=[f'{filename}.npz'] * len(npys_dict[5][i:i + chunk_size]),
                                        eopatch=eopatches[i:i + chunk_size],
                                        patchlet=npys_dict[5][i:i + chunk_size],
                                        chunk_pos=list(range(0, len(eopatches[i:i + chunk_size]))),
                                        timestamp=npys_dict[4][i:i + chunk_size])))

        chunk_counter += 1

    metadata_dir = os.path.dirname(output_dataframe)
    if not os.path.isdir(metadata_dir):
        os.makedirs(metadata_dir)
    
    pd.concat(dfs).to_csv(output_dataframe, index=False)


# Fonction pour la création des .npz
def patchlets_to_npz_files(patchlets_folder: str, output_folder: str, chunk_size: int, output_dataframe: str):
    results = extract_npys_from_folder(patchlets_folder)
    npys_dict = concatenate_npys(results)
    save_into_chunks(output_folder, npys_dict, chunk_size, output_dataframe)


if __name__ == '__main__':
    # Modifier les chemins et autres paramètres selon vos besoins
    patchlets_folder = '/home/joseph/Code/localstorage/patchlets'
    output_folder = '/home/joseph/Code/localstorage/npz_files'
    chunk_size = 1000
    output_dataframe = '/home/joseph/Code/localstorage/dataframe.csv'
    
    patchlets_to_npz_files(patchlets_folder, output_folder, chunk_size, output_dataframe)

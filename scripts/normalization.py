import os
import logging
import sys
import numpy as np
import pandas as pd
from typing import Iterable, Dict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
NPZ_FILES_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/npz_files/')
METADATA_PATH = Path(f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe.csv')
NORMALIZED_METADATA_PATH = Path(
    f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe_normalized.csv')


def stats_per_npz_ts(npz_file_path: str) -> Dict[str, np.array]:
    """
    Calculate various statistics for each time series in an NPZ file.

    Parameters:
        npz_file_path (str): The path to the NPZ file.

    Returns:
        dict: A dictionary containing the calculated statistics for each time series.
            - 'mean': The mean value for each time series.
            - 'median': The median value for each time series.
            - 'perc_1': The 1st percentile value for each time series.
            - 'perc_5': The 5th percentile value for each time series.
            - 'perc_95': The 95th percentile value for each time series.
            - 'perc_99': The 99th percentile value for each time series.
            - 'std': The standard deviation for each time series.
            - 'minimum': The minimum value for each time series.
            - 'maximum': The maximum value for each time series.
            - 'timestamp': The timestamps for each time series.
            - 'patchlet': The patchlets for each time series.
    """
    data = np.load(npz_file_path, allow_pickle=True)
    features = data['X']
    return {
        'mean': np.mean(features, axis=(1, 2)),
        'median': np.median(features, axis=(1, 2)),
        'perc_1': np.percentile(features, q=1, axis=(1, 2)),
        'perc_5': np.percentile(features, q=5, axis=(1, 2)),
        'perc_95': np.percentile(features, q=95, axis=(1, 2)),
        'perc_99': np.percentile(features, q=99, axis=(1, 2)),
        'std': np.std(features, axis=(1, 2)),
        'minimum': np.min(features, axis=(1, 2)),
        'maximum': np.max(features, axis=(1, 2)),
        'timestamp': data['timestamps'],
        'patchlet': data['eopatches']
    }


def create_per_band_norm_dataframe(concatenated_stats: Dict[str, np.array],
                                   stats_keys: Iterable[str],
                                   identifier_keys: Iterable[str]) -> pd.DataFrame:
    """
    Create a pandas DataFrame containing normalized statistics for each band.

    Args:
        concatenated_stats (Dict[str, np.array]): A dictionary containing concatenated statistics.
        stats_keys (Iterable[str]): An iterable of keys representing the statistics.
        identifier_keys (Iterable[str]): An iterable of keys representing the identifiers.

    Returns:
        pd.DataFrame: A pandas DataFrame containing normalized statistics for each band.
    """
    norm_df_dict = {}
    n_bands = concatenated_stats[stats_keys[0]].shape[-1]
    for stat in stats_keys:
        for band in range(0, n_bands):
            norm_df_dict[f'{stat}_b{band}'] = concatenated_stats[stat][..., band]
    for identifier in identifier_keys:
        norm_df_dict[identifier] = concatenated_stats[identifier]
    return pd.DataFrame(norm_df_dict)


def calculate_normalization_factors() -> None:
    """
    Calculate normalization factors for a set of npz files 
    and update a metadata file with the normalization information.

    Returns:
        None
    """
    npz_files = list(Path(NPZ_FILES_DIR).rglob('*.npz'))

    LOGGER.info(f'Compute stats per patchlet for {len(npz_files)} npz files')
    results = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for npz_file in npz_files:
            future = executor.submit(stats_per_npz_ts, npz_file)
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='Computing patchlets statistics'):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                LOGGER.error(f'A task failed: {e}')

    stats_keys = ['mean', 'std', 'median', 'perc_99']
    identifier_keys = ['timestamp', 'patchlet']

    concatenated_stats = {key: np.concatenate(
        [x[key] for x in results]) for key in stats_keys + identifier_keys}

    LOGGER.info('Create dataframe with normalization factors')
    df_stats = create_per_band_norm_dataframe(
        concatenated_stats, stats_keys, identifier_keys)
    df_stats['patchlet'] = df_stats['patchlet'].astype(str)

    # Add month column
    df_stats['timestamp'] = pd.to_datetime(
        df_stats['timestamp'], utc=True).dt.tz_localize(None)
    df_stats['month'] = df_stats['timestamp'].dt.to_period("M")

    # Generate a list of columns for normalization
    norm_cols = [
        f'{stat}_b{band}_mean' for stat in stats_keys for band in range(4)]
    # Generate a list of columns for band statistics aggregation
    aggs = {f'{stat}_b{band}': ['std', 'mean']
            for stat in stats_keys for band in range(4)}

    LOGGER.info('Aggregate normalization stats by month')
    monthly = df_stats.groupby('month', as_index=False).agg(aggs)
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly.rename(columns={'month_': 'month'}, inplace=True)

    LOGGER.info('Add normalization columns to metadata dataframe')
    norm_map = {month: monthly.loc[monthly['month'] == month,
                                   norm_cols].values[0] for month in monthly['month']}
    norm_df = df_stats['month'].map(norm_map).apply(pd.Series)
    norm_df.columns = [
        'norm_perc99_b0', 'norm_perc99_b1', 'norm_perc99_b2',
        'norm_perc99_b3', 'norm_meanstd_mean_b0', 'norm_meanstd_mean_b1',
        'norm_meanstd_mean_b2', 'norm_meanstd_mean_b3', 'norm_meanstd_median_b0',
        'norm_meanstd_median_b1', 'norm_meanstd_median_b2', 'norm_meanstd_median_b3',
        'norm_meanstd_std_b0', 'norm_meanstd_std_b1', 'norm_meanstd_std_b2',
        'norm_meanstd_std_b3'
    ]
    df_stats = pd.concat([df_stats, norm_df], axis=1)

    LOGGER.info(
        f'Add normalization information to metadata file {METADATA_PATH}')
    df_info = pd.read_csv(METADATA_PATH, parse_dates=['timestamp'])
    df_info['timestamp'] = pd.to_datetime(
        df_info['timestamp'], utc=True).dt.tz_localize(None)
    df_info['patchlet'] = df_info['patchlet'].astype(str)
    df_info['month'] = df_info['timestamp'].dt.to_period("M")

    new_df = df_info.merge(df_stats, how='inner', on=[
                           'patchlet', 'month']).reset_index(drop=True)
    LOGGER.info(
        f'Writing normalized metadata file to {NORMALIZED_METADATA_PATH}')
    new_df.to_csv(NORMALIZED_METADATA_PATH, index=False)


if __name__ == '__main__':
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError('NIVA_PROJECT_DATA_ROOT environment variable not set')
    calculate_normalization_factors()

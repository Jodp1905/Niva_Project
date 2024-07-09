import os
import sys
import logging
import numpy as np
import pandas as pd
from functools import partial
from typing import Iterable, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

class ComputeNormalizationConfig:
    def __init__(self, npz_files_folder: str, metadata_file: str):
        self.npz_files_folder = npz_files_folder
        self.metadata_file = metadata_file

def stats_per_npz_ts(npz_file: str, config: ComputeNormalizationConfig) -> Dict[str, np.array]:
    data = np.load(os.path.join(config.npz_files_folder, npz_file), allow_pickle=True)
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

def concat_npz_results(stat: str, results: List[Dict[str, np.array]]) -> np.array:
    return np.concatenate([x[stat] for x in results])

def create_per_band_norm_dataframe(concatenated_stats: Dict[str, np.array], stats_keys: Iterable[str], identifier_keys: Iterable[str]) -> pd.DataFrame:
    norm_df_dict = {}
    n_bands = concatenated_stats[stats_keys[0]].shape[-1]
    for stat in stats_keys:
        for band in range(0, n_bands):
            norm_df_dict[f'{stat}_b{band}'] = concatenated_stats[stat][..., band]
    for identifier in identifier_keys:
        norm_df_dict[identifier] = concatenated_stats[identifier]

    return pd.DataFrame(norm_df_dict)

def calculate_normalization_factors(npz_files_folder: str, metadata_file: str, max_workers: int = 4):
    """ Utility function to calculate normalization factors from the npz files """

    config = ComputeNormalizationConfig(npz_files_folder=npz_files_folder, metadata_file=metadata_file)

    npz_files = [f for f in os.listdir(config.npz_files_folder) if f.endswith('.npz')]

    LOGGER.info('Compute stats per patchlet')
    partial_fn = partial(stats_per_npz_ts, config=config)
    results = [partial_fn(f) for f in npz_files]  # Replaced multiprocess with a simple list comprehension

    stats_keys = ['mean', 'std', 'median', 'perc_99']
    identifier_keys = ['timestamp', 'patchlet'] 

    concatenated_stats = {}
    for key in stats_keys + identifier_keys: 
        concatenated_stats[key] = concat_npz_results(key, results)

    LOGGER.info('Create dataframe with normalization factors')
    df = create_per_band_norm_dataframe(concatenated_stats, stats_keys, identifier_keys)

    # Convert to datetime and ensure no tzinfo
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)

    # Add "month" period
    df['month'] = df['timestamp'].dt.to_period("M")

    norm_cols = [
        norm.format(band) 
        for norm in ['perc_99_b{0}_mean', 'mean_b{0}_mean', 'median_b{0}_mean', 'std_b{0}_mean'] 
        for band in range(4)
    ]

    aggs = {}
    stat_cols = []
    stats = ['perc_99', 'mean', 'median', 'std']
    bands = list(range(4))
    for stat in stats:
        for band in bands:
            aggs[f'{stat}_b{band}'] = [np.std, np.mean]
            stat_cols.append(f'{stat}_b{band}')

    LOGGER.info('Aggregate normalization stats by month')
    monthly = pd.DataFrame(df.groupby('month', as_index=False)[stat_cols].agg(aggs))
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly.rename(columns={'month_': 'month'}, inplace=True)

    def norms(month):
        return monthly.loc[monthly.month == month][norm_cols].values[0]

    df['norm_perc99_b0'], df['norm_perc99_b1'], df['norm_perc99_b2'], df['norm_perc99_b3'], \
    df['norm_meanstd_mean_b0'], df['norm_meanstd_mean_b1'], df['norm_meanstd_mean_b2'], df['norm_meanstd_mean_b3'], \
    df['norm_meanstd_median_b0'], df['norm_meanstd_median_b1'], df['norm_meanstd_median_b2'], df['norm_meanstd_median_b3'], \
    df['norm_meanstd_std_b0'], df['norm_meanstd_std_b1'], df['norm_meanstd_std_b2'], df['norm_meanstd_std_b3'] = zip(*map(norms, df.month))

    LOGGER.info(f'Read metadata file {metadata_file}')
    df_info = pd.read_csv(metadata_file)

    # Ensure the timestamp is datetime with no tzinfo
    df_info['timestamp'] = pd.to_datetime(df_info['timestamp'], utc=True).dt.tz_localize(None)

    LOGGER.info('Add normalization information to metadata file')
    new_df = df_info.merge(df, how='inner', on=['patchlet', 'timestamp'])

    LOGGER.info('Overwrite metadata file with new file')
    new_df.to_csv(metadata_file, index=False)

if __name__ == '__main__':
    npz_files_folder = '/home/joseph/Code/localstorage/npz_files'
    metadata_file = '/home/joseph/Code/localstorage/dataframe.csv' 

    calculate_normalization_factors(npz_files_folder, metadata_file)

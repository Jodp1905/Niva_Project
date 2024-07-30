import warnings
import logging
import fs.move  # required by eopatch.save
import os
import sys
from datetime import datetime, timezone
from xarray import DataArray, Dataset, open_dataset
from eolearn.core import EOPatch, FeatureType, OverwritePermission
from numpy import datetime_as_string, newaxis, concatenate
from sentinelhub.geometry import BBox
from sentinelhub.geometry import CRS
from dateutil.tz import tzlocal
from rasterio import open as rasterio_open, DatasetReader
from rasterio.coords import BoundingBox
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from filter import LogFileFilter

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Suppress DeprecationWarning
warnings.filterwarnings("ignore")

# Define paths
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
NC_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/sentinel2/images/FR/')
MASK_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/sentinel2/masks/FR/')
EOPATCH_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/eopatches/')


def transform_timestamps(time_data: DataArray) -> list:
    """
    Transforms xarray.DataArray timestamps to ISO 8601 strings with utc timezone.

    Args:
        time_data (DataArray): The input xarray.DataArray timestamps.

    Returns:
        list: A list of transformed timestamps as datetime objects with utc timezone.
    """
    time_strings = datetime_as_string(time_data, unit='ms')
    transformed_timestamps = []
    for t in time_strings:
        dt = datetime.strptime(
            t, '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        transformed_timestamps.append(dt)
    return transformed_timestamps


def create_eopatch_from_nc_and_tiff(nc_file_path: str, mask_tiff_path: str,
                                    output_eopatch_dir: str) -> EOPatch:
    """
    Creates an EOPatch object from a NetCDF file and a TIFF mask file.
    Saves the EOPatch in output_eopatch_dir with a unique name.

    Args:
        nc_file_path (str): The path to the NetCDF file.
        mask_tiff_path (str): The path to the TIFF mask file.
        output_eopatch_dir (str): The directory where the EOPatch will be saved.

    Returns:
        int: Status code (0 on success).

    Raises:
        ValueError: If the mask TIFF file does not have 4 bands.

    """
    try:
        LOGGER.debug(f'Entering create_eopatch_from_nc_and_tiff with '
                     f'{nc_file_path}, {mask_tiff_path}, {output_eopatch_dir}')
        ds: Dataset = open_dataset(nc_file_path)
        LOGGER.debug(f'Dataset structure: {ds}')

        # Create a new EOPatch
        eopatch = EOPatch()

        # Add time data to EOPatch
        time_data: DataArray = ds['time'].values
        transformed_timestamps = transform_timestamps(time_data)
        eopatch.timestamp = transformed_timestamps

        # Initialize a dictionary to group bands and other data
        data_dict = {
            'BANDS': [],
            'CLP': None
        }

        LOGGER.debug('Adding raster data from NetCDF to EOPatch')
        for var in ds.data_vars:
            if var != 'spatial_ref':
                data = ds[var].values

                # Eopatch data should have 4 dimensions: time, height, width, channels
                if data.ndim == 2:
                    # Add time and channel dimensions
                    data = data[newaxis, ..., newaxis]
                elif data.ndim == 3:
                    # Add channel dimension
                    data = data[..., newaxis]
                else:
                    LOGGER.warning(
                        f"Variable {var} has unexpected number"
                        f"of dimensions: {data.ndim}. Skipping.")
                    continue

                # Group bands into a single BANDS feature
                if var.startswith('B'):
                    data_dict['BANDS'].append(data)
                elif var == 'CLP':
                    data_dict['CLP'] = data
                else:
                    eopatch.add_feature(FeatureType.DATA, var, data)

        # Stack bands data along the last dimension (channels)
        if data_dict['BANDS']:
            bands_data = concatenate(data_dict['BANDS'], axis=-1)
            eopatch.add_feature(FeatureType.DATA, 'BANDS', bands_data)

        if data_dict['CLP'] is not None:
            eopatch.add_feature(FeatureType.DATA, 'CLP', data_dict['CLP'])

        # Load and add mask features from TIFF
        LOGGER.debug(f'Loading mask data from {mask_tiff_path}')
        with rasterio_open(mask_tiff_path, driver="GTiff") as src:
            src: DatasetReader
            bounds: BoundingBox = src.bounds
            crs_tif = src.crs.to_string()
            bbox: BBox = BBox(bbox=(
                bounds.left, bounds.bottom, bounds.right, bounds.top),
                crs=CRS(crs_tif))
            mask_data = src.read()
            if mask_data.shape[0] != 4:
                raise ValueError(
                    "The mask TIFF should have 4 bands: extent, boundary, distance, enum")

        # Add bbox to EOPatch
        eopatch.bbox = bbox

        # Exract mask data
        # TODO : why are EXTENT, BOUNDARY and ENUM floats ?
        extent = mask_data[0].astype(np.int32)
        boundary = mask_data[1].astype(np.int32)
        distance = mask_data[2]
        enum = mask_data[3].astype(np.int32)

        # Add each tif mask as a feature in the EOPatch
        # TODO : we get a warning here, can we fix it by using FeatureType.DATA_TIMELESS ?
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'EXTENT',
                            extent[..., newaxis])
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'BOUNDARY',
                            boundary[..., newaxis])
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'DISTANCE',
                            distance[..., newaxis])
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'ENUM',
                            enum[..., newaxis])

        # Verify that the features have been added
        for feature in ['EXTENT', 'BOUNDARY', 'DISTANCE', 'ENUM']:
            if feature not in eopatch.mask_timeless:
                LOGGER.error(f'Failed to add {feature} mask to EOPatch')

        # Create and save unique EOPatch
        basename = os.path.basename(nc_file_path)
        image_id = ("_".join(basename.split('_')[:2]))
        eopatch_name = f'eopatch_{image_id}'
        eopatch_path = output_eopatch_dir / eopatch_name
        LOGGER.debug(f'Saving EOPatch to {eopatch_path}')
        os.makedirs(eopatch_path, exist_ok=True)
        eopatch.save(eopatch_path,
                     overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        return 0

    except Exception as e:
        LOGGER.error(f'An error occurred: {e}')
        raise e


def create_all_eopatches() -> None:
    """
    Creates EOPatches from NetCDF and mask files.

    This function creates EOPatches by iterating over pairs of NetCDF and mask files.
    It ensures that the number of NetCDF files and mask files are equal, and that their IDs match.

    Returns:
        None
    """
    EOPATCH_DIR.mkdir(parents=True, exist_ok=True)
    eopatch_dir = EOPATCH_DIR
    nc_files = sorted([nc_path.absolute() for nc_path in NC_DIR.iterdir()
                       if nc_path.is_file()])
    mask_files = sorted([mask_path.absolute() for mask_path in MASK_DIR.iterdir()
                         if mask_path.is_file()])
    assert (len(nc_files) == len(mask_files))
    file_tuples = list(zip(nc_files, mask_files))

    for nc_file, mask_file in file_tuples:
        nc_id = "_".join(nc_file.stem.split('_')[:2])
        mask_id = "_".join(mask_file.stem.split('_')[:2])
        assert (nc_id == mask_id)

    LOGGER.info(
        f'Creating EOPatches from {len(file_tuples)} pairs of NetCDF and mask files')
    with ProcessPoolExecutor() as executor:
        futures = []
        with tqdm(total=len(file_tuples), desc="Creating eopatches") as pbar:
            for nc_file, mask_file in file_tuples:
                future = executor.submit(
                    create_eopatch_from_nc_and_tiff, nc_file, mask_file, eopatch_dir)
                futures.append(future)
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f'A task failed: {e}')
                pbar.update(1)


def main():
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable is not set")
    create_all_eopatches()


if __name__ == "__main__":
    main()

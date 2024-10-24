# At the top of the code, along with the other `import`s
from __future__ import annotations

from tqdm import tqdm
import shutil
import numpy as np
import warnings
import fs.move  # required by eopatch.save even though it is not used directly
import os
import sys
from datetime import datetime, timezone
from netCDF4 import Dataset, num2date
from eolearn.core import EOPatch, FeatureType, OverwritePermission
from numpy import newaxis, concatenate
from sentinelhub.geometry import BBox
from sentinelhub.geometry import CRS
from rasterio import open as rasterio_open, DatasetReader
from rasterio.coords import BoundingBox
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)

# Suppress DeprecationWarning
warnings.filterwarnings("ignore")

# Load configuration
from niva_utils.config_loader import load_config  # noqa: E402
CONFIG = load_config()

# Constants
NIVA_PROJECT_DATA_ROOT = CONFIG['niva_project_data_root']

# Inferred constants
SENTINEL2_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/training_data/sentinel2/')
EOPATCH_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/training_data/eopatches/')
PROCESS_POOL_WORKERS = os.cpu_count()


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

        # Open the NetCDF file using netCDF4.Dataset
        with Dataset(nc_file_path, mode='r') as ds:
            LOGGER.debug(f'Dataset structure: {ds}')

            # Create a new EOPatch
            eopatch = EOPatch()

            # Add time data to EOPatch
            time_data = ds.variables['time'][:]
            time_units = ds.variables['time'].units
            calendar = ds.variables['time'].calendar if hasattr(
                ds.variables['time'], 'calendar') else 'standard'
            transformed_timestamps = num2date(
                time_data, units=time_units, calendar=calendar)
            eopatch.timestamp = [
                datetime(dt.year, dt.month, dt.day, dt.hour,
                         dt.minute, dt.second, tzinfo=timezone.utc)
                for dt in transformed_timestamps]

            # Initialize a dictionary to group bands and other data
            data_dict = {
                'BANDS': [],
                'CLP': None
            }

            LOGGER.debug('Adding raster data from NetCDF to EOPatch')
            for var in ds.variables:
                if var != 'spatial_ref':
                    data: np.ma.MaskedArray = ds.variables[var][:]

                    LOGGER.debug(f"Variable: {var}\n\tshape: {data.shape}\n"
                                 f"\tdtype: {data.dtype}\n\tdata format: {type(data)}\n"
                                 f"\tmask_value: {data.mask}\n\tfill_value: {data.fill_value}"
                                 f"\tnum_invalid: {np.sum(data.mask)}")

                    # Fill numpy.ma.core.MaskedArray with fill_value
                    if data.mask.any():
                        raise ValueError(
                            f"Invalid data found in {var} for file {nc_file_path}")
                    data = data.filled()

                    LOGGER.debug(f"Variable: {var}\n\tshape: {data.shape}\n"
                                 f"\tdtype: {data.dtype}\n\tdata format: {type(data)}\n")

                    # Eopatch data should have 4 dimensions: time, height, width, channels
                    if data.ndim == 2:
                        # Add time and channel dimensions
                        data = data[newaxis, ..., newaxis]
                    elif data.ndim == 3:
                        # Add channel dimension
                        data = data[..., newaxis]
                    else:
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

        # Extract mask data
        extent = mask_data[0].astype(np.int32)
        boundary = mask_data[1].astype(np.int32)
        distance = mask_data[2]
        enum = mask_data[3].astype(np.int32)

        # Add each tif mask as a feature in the EOPatch
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'EXTENT',
                            extent[..., newaxis])
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'BOUNDARY',
                            boundary[..., newaxis])
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'DISTANCE',
                            distance[..., newaxis])
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'ENUM',
                            enum[..., newaxis])

        # Verify that the features have been added
        for feature in ['EXTENT', 'BOUNDARY', 'DISTANCE', 'ENUM']:
            if feature not in eopatch.data_timeless:
                raise ValueError(f'Failed to add {feature} mask to EOPatch')

        # Create and save unique EOPatch
        basename = os.path.basename(nc_file_path)
        image_id = ("_".join(basename.split('_')[:2]))
        eopatch_name = f'eopatch_{image_id}'
        eopatch_path = os.path.join(output_eopatch_dir, eopatch_name)
        LOGGER.debug(f'Saving EOPatch to {eopatch_path}')
        os.makedirs(eopatch_path, exist_ok=True)
        eopatch.save(eopatch_path,
                     overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        return 0

    except Exception as e:
        LOGGER.error(f'An error occurred: {e}', exc_info=True)
        raise e


def create_all_eopatches() -> None:
    """
    Creates EOPatches from NetCDF and mask files.

    This function creates EOPatches by iterating over pairs of NetCDF and mask files.
    It ensures that the number of NetCDF files and mask files are equal, and that their IDs match.

    Raises:
        ValueError: If the number of NetCDF files and mask files do not match,
            or if their IDs do not match.
    Returns:
        None
    """
    EOPATCH_DIR.mkdir(parents=True, exist_ok=True)
    for fold in ["train", "val", "test"]:
        eopatch_dir = EOPATCH_DIR / fold
        if eopatch_dir.exists():
            LOGGER.info(
                f"Detected existing EOPatch directory: {eopatch_dir}, cleaning up")
            shutil.rmtree(eopatch_dir)

        # detect netcdf image files
        eopatch_dir.mkdir(parents=True, exist_ok=True)
        nc_dir = SENTINEL2_DIR / fold / 'images'
        nc_files = sorted([nc_path.absolute() for nc_path in nc_dir.iterdir()
                           if nc_path.is_file()])
        # detect tiff mask files
        mask_dir = SENTINEL2_DIR / fold / 'masks'
        mask_files = sorted([mask_path.absolute() for mask_path in mask_dir.iterdir()
                            if mask_path.is_file()])

        # assert correctness of file pairs
        if len(nc_files) != len(mask_files):
            raise ValueError(
                f"Number of NetCDF files ({len(nc_files)}) "
                f"and mask files ({len(mask_files)}) do not match")
        file_tuples = list(zip(nc_files, mask_files))
        for nc_file, mask_file in file_tuples:
            nc_id = "_".join(nc_file.stem.split('_')[:2])
            mask_id = "_".join(mask_file.stem.split('_')[:2])
            if nc_id != mask_id:
                raise ValueError(
                    f"NetCDF file ID ({nc_id}) does not match mask file ID ({mask_id})")

        LOGGER.info(
            f'Creating EOPatches from {len(file_tuples)} pairs '
            f'of NetCDF and mask files for {fold} fold in {eopatch_dir}')
        # Create EOPatches in parallel
        with ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS) as executor:
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
                        LOGGER.error(f'A task failed: {e}', exc_info=True)
                    pbar.update(1)
        LOGGER.info(f'Created EOPatches for {fold} fold')
    LOGGER.info('All EOPatches created')


def main():
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable is not set")
    create_all_eopatches()


if __name__ == "__main__":
    main()

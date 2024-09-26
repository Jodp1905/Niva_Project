import sys
import os
from datetime import timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd


# TODO delete dependency on sentinelhub
# https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.geometry.html
from sentinelhub.geometry import BBox
# https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.constants.html#sentinelhub.constants.CRS
from sentinelhub.geometry import CRS
import fs.move  # required by eopatch.save
from eolearn.core import EOPatch, FeatureType
from eolearn.core import OverwritePermission


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
TILE_NAME = CONFIG['download_tile']['tile_name']
SPLIT_CONFIG = CONFIG['split_config']

# Inferred constants
TILE_PATH = Path(f"{NIVA_PROJECT_DATA_ROOT}/inference/tile/{TILE_NAME}")
EOPATCHES_FOLDER = Path(f"{NIVA_PROJECT_DATA_ROOT}/inference/eopatches")


def split_save2eopatch(config: dict) -> None:
    # decode all needed to load CRS
    LOGGER.info(f"Loading tile from {config['tile_path']}")
    ds: xr.Dataset = xr.open_dataset(config["tile_path"], decode_coords="all")

    # CRS unit check
    dataset_crs = ds.rio.crs
    if dataset_crs is None:
        raise ValueError(f"No CRS found in the dataset {config['tile_path']}")
    crs_string = dataset_crs.to_string()
    pyproj_crs = CRS(crs_string).pyproj_crs()
    axis_info = pyproj_crs.axis_info
    if len(axis_info) == 0:
        raise ValueError(
            f"CRS axis information is missing for CRS: {crs_string}")
    unit_name = axis_info[0].unit_name
    if unit_name != 'metre':
        raise ValueError(
            f"Expected CRS to have 'metre' as the unit for the first axis, but got {unit_name}")

    # TODO overlap should be somewhere
    # TODO padding from all 4 sides of the tile
    num_split = 0
    for ind_x in range(config["begin_x"], len(ds.x) - config["width"] + 1, config["width"]):
        for ind_y in range(config["begin_y"], len(ds.y) - config["height"] + 1, config["height"]):
            num_split += 1
            patch = ds.isel(x=slice(ind_x, ind_x + config["width"]),
                            y=slice(ind_y, ind_y + config["height"]))
            LOGGER.debug(f"Creating EOPatch {num_split}:\n"
                         f"\tx[{ind_x}:{ind_x + config['width']}]\n"
                         f"\ty[{ind_y}:{ind_y + config['height']}]")
            create_eopatch_ds(
                patch, config["eopatches_folder"], image_id=f"x_{ind_x}_y_{ind_y}_{num_split}")

            if config["num_split"] != -1 and num_split == config["num_split"]:
                break
        if config["num_split"] != -1 and num_split == config["num_split"]:
            break
    LOGGER.info(f"Created {num_split} EOPatches")


def create_eopatch_ds(ds: xr.Dataset,
                      output_eopatch_dir: str,
                      image_id: str = '0') -> EOPatch:
    # according to https://github.com/Jodp1905/Niva_Project/blob/main/scripts/eopatches_for_sampling.py#L61
    # Create a new EOPatch
    eopatch = EOPatch()
    # Add time data to EOPatch
    time_data = ds['time'].values
    transformed_timestamps = [pd.to_datetime(
        t).tz_localize(timezone.utc) for t in time_data]
    eopatch.timestamp = transformed_timestamps

    # Initialize a dictionary to group bands and other data
    data_dict = {
        'BANDS': [],
        'CLP': None
    }

    for var in ds.data_vars:
        if var != 'spatial_ref':
            arr: xr.DataArray = ds[var]
            data: np.ndarray = arr.values
            LOGGER.debug(f"Variable: {var}\n\tshape: {data.shape}\n"
                         f"\tdtype: {data.dtype}\n\tdata format: {type(data)}\n")
            # Eopatch data should have 4 dimensions: time, height, width, channels
            if data.ndim == 2:
                # Add time and channel dimensions
                data = data[np.newaxis, ..., np.newaxis]
            elif data.ndim == 3:
                # Add channel dimension
                data = data[..., np.newaxis]
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
        bands_data = np.concatenate(data_dict['BANDS'], axis=-1)
        eopatch.add_feature(FeatureType.DATA, 'BANDS', bands_data)

    if data_dict['CLP'] is not None:
        eopatch.add_feature(FeatureType.DATA, 'CLP', data_dict['CLP'])

    # Add bbox to EOPatch
    bbox = BBox(ds.rio.bounds(), CRS(ds.rio.crs.to_string()))
    eopatch.bbox = bbox
    # Create and save unique EOPatch
    eopatch_name = f'eopatch_{image_id}'
    eopatch_path = os.path.join(output_eopatch_dir, eopatch_name)
    os.makedirs(eopatch_path, exist_ok=True)
    eopatch.save(eopatch_path,
                 overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return eopatch


def main_subtiles_creation():
    EOPATCHES_FOLDER.mkdir(parents=True, exist_ok=True)
    SPLIT_CONFIG["tile_path"] = TILE_PATH
    SPLIT_CONFIG["eopatches_folder"] = EOPATCHES_FOLDER
    split_save2eopatch(SPLIT_CONFIG)


if __name__ == "__main__":
    main_subtiles_creation()

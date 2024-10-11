import sys
import os
from datetime import timezone, datetime
import xarray as xr
from pathlib import Path
import numpy as np
import itertools as it
from typing import Any, Iterable, cast


# TODO delete dependency on sentinelhub
# https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.geometry.html
from sentinelhub.geometry import BBox
# https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.constants.html#sentinelhub.constants.CRS
from sentinelhub.geometry import CRS
import fs.move  # required by eopatch.save
from eolearn.core import EOPatch, FeatureType
from eolearn.core import OverwritePermission
from tqdm import tqdm

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
SPLIT_CONFIG = CONFIG['split_config']

def check_good_patch(patch, bad_percentage=0.70, vgt_percentage=0.1,
                     cld_percentage=0.15):
    # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    shape_all = patch.scl.shape[-1] * patch.scl.shape[-2]
    cld_data_count, bad_data_count = 0, 0
    cld_data_count += (patch.scl == 9).sum()  # Cloud high probability
    bad_data_count += cld_data_count.values
    bad_data_count += (patch.scl == 11).sum()  # Snow or ice
    bad_data_count += (patch.scl == 0).sum()  # No Data (Missing data)
    bad_data_count += (patch.scl == 1).sum()  # Saturated or defective pixel
    bad_data_count += (patch.scl == 5).sum()  # Not-vegetated
    bad_data_count += (patch.scl == 6).sum()  # Water

    # cld_data_count += (patch.scl == 3).sum()  # Cloud shadows
    cld_data_count += (patch.scl == 8).sum()  # Cloud medium probability
    # cld_data_count += (patch.scl == 10).sum()  # Thin cirrus
    # vegetation (possible crop fields). Problem - could be under the clouds / cloud shadows ... -> another class
    vegetation_prec = (patch.scl == 4).sum().values / shape_all  # Vegetation
    curr_bad_percentage = bad_data_count.values / shape_all
    curr_cld_percentage = cld_data_count.values / shape_all
    flag_good = (curr_cld_percentage < cld_percentage) & ((curr_bad_percentage < bad_percentage) | (vegetation_prec > vgt_percentage))
    # True - good patch, False - bad
    return flag_good

def split_save2eopatch(config: dict) -> None:
    # decode all needed to load CRS
    LOGGER.info(f"Loading tile from {config['tile_path']}")
    dataset: xr.Dataset = xr.open_dataset(config["tile_path"], decode_coords="all")

    # CRS unit check
    assert (
            CRS(dataset.rio.crs.to_string()).pyproj_crs().axis_info[0].unit_name == "metre"
    ), "The resulting CRS should have axis units in metres."

    W, H = len(dataset.x), len(dataset.y)
    begin_x, begin_y, width, height, overlap = (config["begin_x"], config["begin_y"],
                                                config["width"], config["height"],
                                                config["overlap"])

    x_rng = range(begin_x, W - width + 1, width - overlap)
    y_rng = range(begin_y, H - height + 1, height - overlap)

    xy_index = list(it.product(x_rng, y_rng))  # small error x and x

    if (W - width) % (width - overlap) != 0:
        xy_index.extend((W - width, y) for y in y_rng)

    if (H - height) % (height - overlap) != 0:
        xy_index.extend((x, H - height) for x in x_rng)

    # bottom right corner if needed
    if (W - width) % (width - overlap) != 0 and (H - height) % (height - overlap) != 0:
        xy_index.append((W - width, H - height))

    num_bad = 0
    num_split = 0
    for ind_x, ind_y in tqdm(xy_index,
                             desc="Splitting and saving to EOPatch"):
        patch = dataset.isel(x=slice(ind_x, ind_x + config["width"]),
                        y=slice(ind_y, ind_y + config["height"]))
        if check_good_patch(patch, config["bad_percentage"],
                            config["vgt_percentage"],
                            config["cld_percentage"]):
            num_split += 1
            create_eopatch_ds(patch, config["eopatches_folder"],
                              image_id=f"x_{ind_x}_y_{ind_y}_{num_split}")
        else:
            num_bad += 1
        if config["num_split"] != -1 and num_split == config["num_split"]:
            break
    LOGGER.info(f"Number of bad patches = {num_bad} from all {len(xy_index)}")
    LOGGER.info(f"Created {num_split} EOPatches")


def create_eopatch_ds(ds: xr.Dataset,
                      output_eopatch_dir: str,
                      image_id: str = '0') -> EOPatch:
    # according to https://github.com/Jodp1905/Niva_Project/blob/main/scripts/eopatches_for_sampling.py#L61
    # Create a new EOPatch
    eopatch = EOPatch(
        bbox=BBox(ds.rio.bounds(), CRS(ds.rio.crs.to_string())),
        timestamp=[
            dt.astimezone(timezone.utc)
            for dt in cast(
                list[datetime],
                ds["time"].values.astype("datetime64[ms]").tolist(),  # type: ignore
            )
        ],
    )

    # Initialize a dictionary to group bands and other data
    data_dict = {
        'BANDS': [],
        'CLP': None
    }

    for var in ds.data_vars:
        if var != 'spatial_ref':
            arr: xr.DataArray = ds[var]
            data: np.ndarray = arr.values
            # LOGGER.debug(f"Variable: {var}\n\tshape: {data.shape}\n"
            #              f"\tdtype: {data.dtype}\n\tdata format: {type(data)}\n")
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


    # Create and save unique EOPatch
    eopatch_name = f'eopatch_{image_id}'
    eopatch_path = os.path.join(output_eopatch_dir, eopatch_name)
    os.makedirs(eopatch_path, exist_ok=True)
    eopatch.save(eopatch_path,
                 overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return eopatch


def main_split_save2eopatch(TILE_PATH, EOPATCHES_FOLDER):
    SPLIT_CONFIG["tile_path"] = TILE_PATH
    SPLIT_CONFIG["eopatches_folder"] = EOPATCHES_FOLDER
    split_save2eopatch(SPLIT_CONFIG)


if __name__ == "__main__":
    PROJECT_DATA_ROOT = CONFIG['niva_project_data_root_inf']
    TILE_ID = CONFIG['TILE_ID']
    # Inferred constants
    PROJECT_DATA_ROOT = os.path.join(PROJECT_DATA_ROOT, TILE_ID)
    TILE_PATH = os.path.join(PROJECT_DATA_ROOT, "tile",
                             f"{TILE_ID}.nc")  # here should be the tile saved during download
    # the folders that will be created during the pipeline run
    EOPATCHES_FOLDER = os.path.join(PROJECT_DATA_ROOT, "eopatches")
    main_split_save2eopatch(TILE_PATH, EOPATCHES_FOLDER)

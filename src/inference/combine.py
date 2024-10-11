from skimage.morphology import disk
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from eolearn.io import ExportToTiffTask
from eolearn.core import EOPatch, FeatureType
from eolearn.core import OverwritePermission
from PIL import Image
from skimage.filters import rank
from pathlib import Path
import fs.move  # required by eopatch.save

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
COMBINE_CONFIG = CONFIG['combine_config']


def smooth(array: np.ndarray, disk_size: int = 2) -> np.ndarray:
    # taken from https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py
    """ Blur input array using a disk element of a given disk size """
    assert array.ndim == 2

    smoothed = rank.mean(array, selem=disk(
        disk_size).astype(np.float32)).astype(np.float32)
    smoothed = (smoothed - np.min(smoothed)) / \
        (np.max(smoothed) - np.min(smoothed))

    assert np.sum(~np.isfinite(smoothed)) == 0

    return smoothed


def upscale_and_rescale(array: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    # taken from https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py
    """ Upscale a given array by a given scale factor using bicubic interpolation """
    assert array.ndim == 2

    height, width = array.shape

    rescaled = np.array(Image.fromarray(array).resize(
        (width * scale_factor, height * scale_factor), Image.BICUBIC))
    rescaled = (rescaled - np.min(rescaled)) / \
        (np.max(rescaled) - np.min(rescaled))

    assert np.sum(~np.isfinite(rescaled)) == 0

    return rescaled


def compile_up_sample(eopatch, tidx=0, disk_size=2, scale_factor=2):
    # taken from https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py
    """ Run merging and upscaling """
    extent = eopatch.data["EXTENT_PREDICTED"][tidx, ..., 0]
    boundary = eopatch.data["BOUNDARY_PREDICTED"][tidx, ..., 0]

    combined = np.clip(1 + extent - boundary, 0, 2)

    combined = upscale_and_rescale(combined, scale_factor=scale_factor)
    combined = smooth(combined, disk_size=disk_size)
    combined = upscale_and_rescale(combined, scale_factor=scale_factor)
    array = smooth(combined, disk_size=disk_size * scale_factor)

    eopatch.add_feature(FeatureType.DATA_TIMELESS,
                        f'COMBINED_PREDICTED', np.expand_dims(array, axis=-1))
    return eopatch


def save2tiff(eop_path, tiffs_folder, tidx=0, disk_size=2, scale_factor=2):
    eopatch = EOPatch.load(eop_path)
    eopatch = compile_up_sample(eopatch, tidx=tidx, disk_size=disk_size, scale_factor=scale_factor)
    eopatch.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    filename = os.path.join(tiffs_folder, f'{os.path.split(eop_path)[1]}-{eopatch.bbox.crs.epsg}.tiff')
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py#L248
    export_task = ExportToTiffTask(feature=(FeatureType.DATA_TIMELESS, f'COMBINED_PREDICTED'),
                                   folder=".",
                                   compress='DEFLATE', image_dtype=np.float32)
    export_task.execute(eopatch, filename=filename)
    return eopatch

def run_combine(config):

    eopatches = [f.path for f in os.scandir(
        config["eopatches_folder"]) if f.is_dir() and f.name.startswith('eopatch')]

    LOGGER.info(f'Combine extent & boundary and save to tiff for {len(eopatches)} EOPatch files')
    results = []
    # TODO delete time_interval from combine and vectorization
    tiffs_folder = config["tiffs_folder"]
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        futures = []
        for file_path in eopatches:
            future = executor.submit(save2tiff, eop_path=file_path,
                                     tiffs_folder=tiffs_folder, tidx=0,
                                     disk_size=config["disk_size"], scale_factor=config["scale_factor"])
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='Combine and save to tiff predictions'):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                LOGGER.error(f'A task failed: {e}')
    return results



def main_combine(EOPATCHES_FOLDER, PREDICTIONS_DIR):
    COMBINE_CONFIG["eopatches_folder"] = EOPATCHES_FOLDER
    COMBINE_CONFIG["tiffs_folder"] = PREDICTIONS_DIR
    res = run_combine(COMBINE_CONFIG)
    LOGGER.info(f'Combine and save to tiff completed for {len(res)} EOPatches')


if __name__ == "__main__":
    # Constants
    PROJECT_DATA_ROOT = CONFIG['niva_project_data_root_inf']
    TILE_ID = CONFIG['TILE_ID']
    # Inferred constants
    PROJECT_DATA_ROOT = os.path.join(PROJECT_DATA_ROOT, TILE_ID)
    # the folders that will be created during the pipeline run
    EOPATCHES_FOLDER = os.path.join(PROJECT_DATA_ROOT, "eopatches")
    PREDICTIONS_DIR = os.path.join(PROJECT_DATA_ROOT, "predictions")
    main_combine(EOPATCHES_FOLDER, PREDICTIONS_DIR)

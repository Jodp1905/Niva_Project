from create_subtiles import main_split_save2eopatch
from predict import main_prediction
from combine import main_combine
from vectorisation import main_vectorisation
from post_process import main_postprocess
import os
import sys
from time import time

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
PROJECT_DATA_ROOT = CONFIG['niva_project_data_root_inf']
TILE_ID = CONFIG['TILE_ID']

# Inferred constants
PROJECT_DATA_ROOT = os.path.join(PROJECT_DATA_ROOT, TILE_ID)
TILE_PATH = os.path.join(PROJECT_DATA_ROOT, "tile", f"{TILE_ID}.nc")  # here should be the tile saved during download
# the folders that will be created during the pipeline run
EOPATCHES_FOLDER = os.path.join(PROJECT_DATA_ROOT, "eopatches")
PREDICTIONS_DIR = os.path.join(PROJECT_DATA_ROOT, "predictions")
CONTOURS_DIR = os.path.join(PROJECT_DATA_ROOT, "contours")
GPKG_FILE_PATH = os.path.join(CONTOURS_DIR, f"{TILE_ID}.gpkg")


if __name__ == "__main__":
    durations = []

    # split tile to sub-tiles filter and convert to EOPatch
    start_time = time()
    main_split_save2eopatch(TILE_PATH, EOPATCHES_FOLDER)
    end_time = time()
    durations.append(('convert_to_eopatches', end_time - start_time))

    # Predict by ResUnet-a model on EOPatches.
    start_time = time()
    main_prediction(EOPATCHES_FOLDER)
    end_time = time()
    durations.append(('predict_all_eopatches', end_time - start_time))

    # Combine predicted boundary and extent of EOPatch and save to tiff
    start_time = time()
    main_combine(EOPATCHES_FOLDER, PREDICTIONS_DIR)
    end_time = time()
    durations.append(('combine_prediction_output', end_time - start_time))

    # Vectorize tiffs from previous step
    # TODO check if gdal operations supports GPU
    # The longest step
    start_time = time()
    main_vectorisation(GPKG_FILE_PATH, PROJECT_DATA_ROOT, PREDICTIONS_DIR, CONTOURS_DIR)
    end_time = time()
    durations.append(('vectorization', end_time - start_time))

    # Final Post-process of GeoPackage vector data generate from previous step
    # (simplify, filter small/large polygons, save to geojson)
    start_time = time()
    main_postprocess(GPKG_FILE_PATH)
    end_time = time()
    durations.append(('final_post_process', end_time - start_time))

    LOGGER.info("\n--- Overview ---")
    total_time = 0
    for step, duration in durations:
        LOGGER.info(f"Total time for {step}: {duration:.2f} seconds")
        total_time += duration
    LOGGER.info(f"Total time for all steps: {total_time:.2f} seconds")

from create_subtiles import main_subtiles_creation
from predict import main_prediction
from combine import main_combine
from vectorisation import main_vectorisation
from post_process import main_postprocess
from pathlib import Path
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
NIVA_PROJECT_DATA_ROOT = CONFIG['niva_project_data_root']
MODEL_FOLDER = Path("checkpoints")
MODEL_CONFIG_PATH = Path("model_cfg.json")  # folder with model's config file

# Inferred constants
TILE_PATH = Path(f"{NIVA_PROJECT_DATA_ROOT}/inference/tile/input_tile.nc")
EOPATCHES_FOLDER = Path(f"{NIVA_PROJECT_DATA_ROOT}/inference/eopatches")
RASTER_RESULTS_FOLDER = Path(f"{NIVA_PROJECT_DATA_ROOT}/inference/tiffs")
PREDICTIONS_DIR = Path(f"{NIVA_PROJECT_DATA_ROOT}/predictions")
CONTOURS_DIR = Path(f"{NIVA_PROJECT_DATA_ROOT}/contours")
PROCESS_POOL_WORKERS = os.cpu_count()

if __name__ == "__main__":
    durations = []

    # TODO taking into consideration vegetation, missing / cloudy data, split with overlap, split with padding
    # split tile to sub-tiles and convert to EOPatch
    start_time = time()
    main_subtiles_creation()
    end_time = time()
    durations.append(('convert_to_eopatches', end_time - start_time))

    # Predict by ResUnet-a model on EOPatches.
    # TODO GPU here with batch inferencing, ONNX converted ?
    start_time = time()
    main_prediction()
    end_time = time()
    durations.append(('predict_all_eopatches', end_time - start_time))

    # Combine predicted boundary and extent of EOPatch and save to tiff
    start_time = time()
    main_combine()
    end_time = time()
    durations.append(('combine_prediction_output', end_time - start_time))

    # Vectorize tiffs from previous step
    # TODO check if gdal operations supports GPU
    # The longest step
    start_time = time()
    main_vectorisation()
    end_time = time()
    durations.append(('vectorization', end_time - start_time))

    # Final Post-process of GeoPackage vector data generate from previous step
    # (simplify, filter small/large polygons, save to geojson)
    start_time = time()
    # Created 92,529 records from one tile - 100 sub-tiles (without padding)
    main_postprocess()
    end_time = time()
    durations.append(('final_post_process', end_time - start_time))

    LOGGER.info("\n--- Overview ---")
    total_time = 0
    for step, duration in durations:
        LOGGER.info(f"Total time for {step}: {duration:.2f} seconds")
        total_time += duration
    LOGGER.info(f"Total time for all steps: {total_time:.2f} seconds")

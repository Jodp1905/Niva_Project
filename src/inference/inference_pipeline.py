from pathlib import Path
import os
import logging
from time import time
import warnings
warnings.filterwarnings("ignore")

from utils_pipeline import run_predict, run_combine, run_vectorize, split_save2eopatch, \
    final_post_process

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

PROJECT_DATA_ROOT = ...  # paste here your path
TILE_PATH = os.path.join(PROJECT_DATA_ROOT, "tile", "S2B_31TEN_20230301_0_L2A.nc")  # here should be the tile saved during download
# paths to checkpoints folder and configuration folder -> change to your own
MODEL_FOLDER = Path("checkpoints") 
MODEL_CONFIG_PATH = Path("model_cfg.json")  # folder with model's config file 

# the folders that will be created during the pipeline run
EOPATCHES_FOLDER = os.path.join(PROJECT_DATA_ROOT, "eopatches")
RASTER_RESULTS_FOLDER = os.path.join(PROJECT_DATA_ROOT, "tiffs")
PREDICTIONS_DIR = os.path.join(PROJECT_DATA_ROOT, "p")
CONTOURS_DIR = os.path.join(PROJECT_DATA_ROOT, "c")


PROCESS_POOL_WORKERS = int(os.getenv('PROCESS_POOL_WORKERS', os.cpu_count()))  # 12

split_config = {
        "tile_path": TILE_PATH,
        "eopatches_folder": EOPATCHES_FOLDER,
        "height": 1000,
        "width": 1000,
        "overlap": 0,
        "num_split": -1,  # all sub-tile splits with parameter -1
        "begin_x": 0,
        "begin_y": 0,
}

prediction_config = {
        "eopatches_folder": EOPATCHES_FOLDER,
        "model_path": MODEL_FOLDER,  # the ResUnet-a model can handle dynamic shapes
        "height": 1024,  # eopatch height + 2 pad_buffer, should by div by 32 filter of model
        "width": 1024,  # eopatch width + 2 pad_buffer, should by div by 32 filter of model
        "pad_buffer": 12,
        "crop_buffer": 12,
        "n_channels": 4,
        "model_cfg_path": MODEL_CONFIG_PATH,
        "batch_size": 1
}

combine_config = {
    "eopatches_folder": EOPATCHES_FOLDER,
    "tiffs_folder": RASTER_RESULTS_FOLDER,
    "scale_factor": 2,
    "disk_size": 2,
    "workers": PROCESS_POOL_WORKERS,
    # "time_interval": "0",  # TODO delete time_interval from combine and vectorization
}

vectorize_config = {
        "tiffs_folder": RASTER_RESULTS_FOLDER,
        # "time_intervals": ["0"],  # change to your month of the tile # TODO delete time_interval from combine and vectorization
        # "utms": ["32631"],  # List all the different UTM zones within the AOI = Tile
        "shape": [4000, 4000],  # scale_factor * scale_factor=2 * EOPatch shape = 1000
        "buffer": [200, 200],
        "weights_file": os.path.join(PROJECT_DATA_ROOT, "weights.tiff"),  # creates file during vectorization
        "vrt_dir": PROJECT_DATA_ROOT,
        "predictions_dir": PREDICTIONS_DIR,
        "contours_dir": CONTOURS_DIR,
        "max_workers": 8,
        "chunk_size": 500,
        "chunk_overlap": 10,
        "threshold": 0.6,
        "cleanup": True,
        "skip_existing": True,
        "rows_merging": True
}

final_config = {
    "gpkg_path": os.path.join(CONTOURS_DIR, "merged_0_32631.gpkg"),  # TODO filename same in vectorize / final
    "simplify_tolerance": 2.5,
    "smallest_area": 2,
    "biggest_area": 10**7,
    "version": 1.3,
}


if __name__ == "__main__":
    durations = []

    # TODO taking into consideration vegetation, missing / cloudy data, split with overlap, split with padding
    # split tile to sub-tiles and convert to EOPatch
    start_time = time()
    split_save2eopatch(split_config)
    end_time = time()
    durations.append(('convert_to_eopatches', end_time - start_time))

    # Predict by ResUnet-a model on EOPatches.
    # TODO GPU here with batch inferencing, ONNX converted ?
    start_time = time()
    results = run_predict(prediction_config)
    end_time = time()
    durations.append(('predict_all_eopatches', end_time - start_time))

    # Combine predicted boundary and extent of EOPatch and save to tiff
    start_time = time()
    results = run_combine(combine_config)
    end_time = time()
    durations.append(('combine_prediction_output', end_time - start_time))

    # Vectorize tiffs from previous step
    # TODO check if gdal operations supports GPU
    # The longest step
    start_time = time()
    results = run_vectorize(vectorize_config)
    end_time = time()
    durations.append(('vectorization', end_time - start_time))

    # Final Post-process of GeoPackage vector data generate from previous step
    # (simplify, filter small/large polygons, save to geojson)
    start_time = time()
    results = final_post_process(final_config)  # Created 92,529 records from one tile - 100 sub-tiles (without padding)
    end_time = time()
    durations.append(('final_post_process', end_time - start_time))

    logging.info("\n--- Overview ---")
    total_time = 0
    for step, duration in durations:
        logging.info(f"Total time for {step}: {duration:.2f} seconds")
        total_time += duration
    logging.info(f"Total time for all steps: {total_time:.2f} seconds")

from create_eopatches import create_all_eopatches
from create_npz import eopatches_to_npz_files
from create_datasets import create_datasets
from niva_utils.logger import get_logger
import logging
import time
from pathlib import Path

# Import logger
LOGGER = get_logger(__name__)

# Import config
from niva_utils.config_loader import load_config  # noqa: E402
CONFIG = load_config()

# Paths parameters
NIVA_PROJECT_DATA_ROOT = CONFIG['niva_project_data_root']
SENTINEL2_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/training_data/sentinel2/')


def main():
    """
    Main function to preprocess data for the Niva Project.

    This function performs the following steps:
    1. Creates eopatches from the NetCDF sentinel-2 images and tiff masks.
    2. Converts eopatches to binary numpy files (npz).
    3. Creates normalized and augmented datasets from the npz files.

    """
    if not NIVA_PROJECT_DATA_ROOT:
        LOGGER.error(
            "Niva project data root is not set in the configuration file.")
        exit(1)

    if not SENTINEL2_DIR.exists():
        LOGGER.error(
            f"Sentinel-2 data directory {SENTINEL2_DIR} does not exist."
            f"Please download the data using the download_data.sh script.")
        exit(1)

    durations = []

    try:
        start_time = time.time()
        create_all_eopatches()
        end_time = time.time()
        durations.append(('create_all_eopatches', end_time - start_time))

        # SKIPPED : Patchlets creation is incompatible with resunet-a 6d model training
        # because eopatches are already of size 256, the minimum size for the model

        start_time = time.time()
        eopatches_to_npz_files()
        end_time = time.time()
        durations.append(('patchlets_to_npz_files', end_time - start_time))

        # SKIPPED : Normalization has been simplified and this step is no longer required

        start_time = time.time()
        create_datasets()
        end_time = time.time()
        durations.append(('create_datasets', end_time - start_time))

        logging.info(
            "Preprocessing completed successfully, data is ready for training.")
        logging.info("\n--- Overview ---")
        total_time = 0
        for step, duration in durations:
            logging.info(f"Total time for {step}: {duration:.2f} seconds")
            total_time += duration
        logging.info(f"Total time for all steps: {total_time:.2f} seconds")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()

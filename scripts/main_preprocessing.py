from eopatches_for_sampling import create_all_eopatches
from patchlets_to_npz import patchlets_to_npz_files
from create_datasets import create_datasets
from data_split import main_download
from os import getenv
import logging
import time
import sys

from filter import LogFileFilter

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

NIVA_PROJECT_DATA_ROOT = getenv('NIVA_PROJECT_DATA_ROOT')
DOWNLOAD_DATA = 0


def main():
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable is not set")

    durations = []

    try:
        if DOWNLOAD_DATA:
            start_time = time.time()
            main_download()
            end_time = time.time()
            durations.append(('main_download', end_time - start_time))

        start_time = time.time()
        create_all_eopatches()
        end_time = time.time()
        durations.append(('create_all_eopatches', end_time - start_time))

        # SKIPPED : Patchlets creation is incompatible with resunet-a 6d model training
        # because eopatches are already of size 256, the minimum size for the model

        start_time = time.time()
        patchlets_to_npz_files()
        end_time = time.time()
        durations.append(('patchlets_to_npz_files', end_time - start_time))

        # SKIPPED : Normalization has been simplified and this step is no longer required

        start_time = time.time()
        create_datasets()
        end_time = time.time()
        durations.append(('create_datasets', end_time - start_time))

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

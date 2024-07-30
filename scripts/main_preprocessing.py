from eopatches_for_sampling import create_all_eopatches
from creation_patchlets import create_all_patchlets
from patchlets_to_npz import patchlets_to_npz_files
from normalization import calculate_normalization_factors
from create_kfold import k_folds
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


def main():
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable is not set")

    durations = []

    try:
        start_time = time.time()
        create_all_eopatches()
        end_time = time.time()
        durations.append(('create_all_eopatches', end_time - start_time))

        start_time = time.time()
        patchlets_to_npz_files()
        end_time = time.time()
        durations.append(('patchlets_to_npz_files', end_time - start_time))

        start_time = time.time()
        calculate_normalization_factors()
        end_time = time.time()
        durations.append(
            ('calculate_normalization_factors', end_time - start_time))

        start_time = time.time()
        k_folds()
        end_time = time.time()
        durations.append(('k_folds', end_time - start_time))

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

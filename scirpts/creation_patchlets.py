import os
import logging
import numpy as np
from typing import List, Tuple
from eolearn.core import EOPatch, FeatureType, FeatureTypeSet, EOTask, OverwritePermission
import fs.move
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Suppress DeprecationWarning
warnings.filterwarnings("ignore")

# Define paths
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
EOPATCH_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/eopatches/')
PATCHLETS_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/patchlets/')

# Define EOTask parameters
EOTASK_SAMPLE_POSITIVE = os.getenv('SAMPLE_POSITIVE', "True").lower() == "true"
EOTASK_MASK_FEATURE_NAME = os.getenv('MASK_FEATURE_NAME', 'EXTENT')
EOTASK_BUFFER = os.getenv('BUFFER', 0)
EOTASK_PATCH_SIZE = os.getenv('PATCH_SIZE', 32)
EOTASK_NUM_SAMPLES = os.getenv('NUM_SAMPLES', 20)
EOTASK_MAX_RETRIES = os.getenv('MAX_RETRIES', 100)
EOTASK_FRACTION_VALID = os.getenv('FRACTION_VALID', 0.4)
EOTASK_SAMPLED_FEATURE_NAME = os.getenv('SAMPLED_FEATURE_NAME', 'BANDS')


class SamplePatchlets(EOTask):
    """
    Task for sampling patchlets from a reference mask.

    :param feature: A tuple containing the feature type and name of the reference mask.
    :type feature: Tuple[FeatureType, str]
    :param buffer: The buffer around the patchlets to avoid edge effects.
    :type buffer: int
    :param patch_size: The size of the patchlets to sample.
    :type patch_size: int
    :param num_samples: The number of patchlets to sample.
    :type num_samples: int
    :param max_retries: The maximum number of retries to sample a patchlet.
    :type max_retries: int
    :param sample_features: A list of tuples containing the feature type and name 
    of the features to sample.
    :type sample_features: List[Tuple[FeatureType, str]]
    :param fraction_valid: The fraction of valid (or invalid) data in the patchlets.
    :type fraction_valid: float
    :param no_data_value: The no data value in the reference mask.
    :type no_data_value: int
    :param sample_positive: A flag to sample patchlets with a positive condition.
    :type sample_positive: bool
    """

    def __init__(self, feature: Tuple[FeatureType, str],
                 buffer: int,
                 patch_size: int,
                 num_samples: int,
                 max_retries: int,
                 sample_features: List[Tuple[FeatureType, str]],
                 fraction_valid: float = 0.2,
                 no_data_value: int = 0,
                 sample_positive: bool = True):
        self.feature_type, self.feature_name, self.new_feature_name = next(
            self._parse_features(feature, new_names=True,
                                 default_feature_type=FeatureType.MASK_TIMELESS,
                                 allowed_feature_types={
                                     FeatureType.MASK_TIMELESS},
                                 rename_function='{}_SAMPLED'.format)())
        self.max_retries = max_retries
        self.fraction = fraction_valid
        self.no_data_value = no_data_value
        self.sample_features = self._parse_features(sample_features)
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.buffer = buffer
        self.sample_positive = sample_positive

    def _area_fraction_condition(self, ratio: float) -> bool:
        """
        Check if the given ratio satisfies the area fraction condition.
        Results depends on the sample_positive flag.

        Args:
            ratio (float): The ratio to be checked.

        Returns:
            bool: True if the ratio satisfies the condition, False otherwise.
        """
        return ratio < self.fraction if self.sample_positive else ratio > self.fraction

    def execute(self, eopatch: EOPatch, seed: int = None) -> List[EOPatch]:
        """
        Executes the creation of patchlets from the given EOPatch.

        Args:
            eopatch (EOPatch): The input EOPatch from which patchlets will be created.
            seed (int, optional): The seed value for random number generation. Defaults to None.

        Returns:
            List[EOPatch]: A list of EOPatch objects representing the created patchlets.
        """
        mask = eopatch[self.feature_type][self.feature_name].squeeze()

        # Convert mask to int32 to avoid DeprecationWarning
        if mask.dtype != np.int32 and mask.dtype != bool:
            mask = mask.astype(np.int32)

        # Ensure mask is 3-dimensional
        # TODO : why add a new axis when we just squeezed the 3D "EXTENT" mask ?
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]

        n_rows, n_cols, _ = mask.shape

        if mask.ndim != 3:
            raise ValueError('Invalid shape of sampling reference map.')

        # Check if the patch size and buffer are valid
        if self.patch_size > n_rows or self.patch_size > n_cols:
            raise ValueError(
                f"Patch size {self.patch_size} is too large "
                f"for the mask dimensions {n_rows}x{n_cols}.")

        np.random.seed(seed)
        eops_out = []

        # TODO : there is no colision check between patchlets
        for patchlet_num in range(0, self.num_samples):
            ratio = 0.0 if self.sample_positive else 1
            retry_count = 0
            new_eopatch = EOPatch(timestamp=eopatch.timestamp)

            while self._area_fraction_condition(ratio) and retry_count < self.max_retries:
                if n_rows - self.patch_size - self.buffer <= 0 \
                        or n_cols - self.patch_size - self.buffer <= 0:
                    LOGGER.warning(
                        f"The patch size {self.patch_size} and buffer {self.buffer} "
                        f"are too large for the mask dimensions {n_rows}x{n_cols}."
                        f" Trying without buffer.")
                    row = 0
                    col = 0
                else:
                    row = np.random.randint(
                        self.buffer, n_rows - self.patch_size - self.buffer)
                    col = np.random.randint(
                        self.buffer, n_cols - self.patch_size - self.buffer)
                patchlet = mask[row:row + self.patch_size,
                                col:col + self.patch_size]
                ratio = np.sum(patchlet != self.no_data_value)  \
                    / self.patch_size ** 2
                retry_count += 1

            if retry_count == self.max_retries:
                LOGGER.debug(f'Could not determine an area with good enough ratio '
                             f'of valid sampled pixels for '
                             f'patchlet number: {patchlet_num}')
                continue

            for feature_type, feature_name in self.sample_features:
                # TODO : this filter removes NVDI, bbox, and ENUM
                # so why carry them in the first place ?
                if feature_type in FeatureTypeSet.RASTER_TYPES \
                        .intersection(FeatureTypeSet.SPATIAL_TYPES):
                    feature_data = eopatch[feature_type][feature_name]
                    if feature_type.is_time_dependent():
                        sampled_data = feature_data[:,
                                                    row:row + self.patch_size,
                                                    col:col + self.patch_size, :]
                    else:
                        sampled_data = feature_data[row:row + self.patch_size,
                                                    col:col + self.patch_size, :]

                    # Add the sampled data to the new EOPatch
                    new_eopatch.add_feature(
                        feature_type, feature_name, sampled_data)

            eops_out.append(new_eopatch)

        return eops_out


def create_patchlets(eopatch_folder_path: str) -> None:
    """
    Create patchlets from an EOPatch.
    All creation parameters can be set as environment variables.

    Args:
        eopatch_folder_path (str): The path to the folder containing the EOPatch.

    Returns:
        int : The number of patchlets created. May be less than the requested number.
    """
    eopatch = EOPatch.load(eopatch_folder_path, lazy_loading=True)
    task = SamplePatchlets(
        feature=(FeatureType.MASK_TIMELESS, EOTASK_MASK_FEATURE_NAME),
        buffer=EOTASK_BUFFER,
        patch_size=EOTASK_PATCH_SIZE,
        num_samples=EOTASK_NUM_SAMPLES,
        max_retries=EOTASK_MAX_RETRIES,
        fraction_valid=EOTASK_FRACTION_VALID,
        sample_features=[
            (FeatureType.DATA, EOTASK_SAMPLED_FEATURE_NAME),
            (FeatureType.MASK_TIMELESS, 'EXTENT'),
            (FeatureType.MASK_TIMELESS, 'BOUNDARY'),
            (FeatureType.MASK_TIMELESS, 'DISTANCE')
        ],
        sample_positive=EOTASK_SAMPLE_POSITIVE
    )
    patchlets = task.execute(eopatch)
    basename = os.path.basename(eopatch_folder_path)
    for i, patchlet in enumerate(patchlets):
        patchlet_path = f"{PATCHLETS_DIR}/patchlet_{basename}_{i+1}"
        patchlet.save(patchlet_path,
                      overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return len(patchlets)


def create_all_patchlets():
    """
    Creates patchlets for all EOPatches in the EOPATCH_DIR directory.

    This function resets the patchlets directory, deletes existing patchlets, 
    and creates new patchlets for each EOPatch in the EOPATCH_DIR directory. 
    The patchlets are created in parallel using a ProcessPoolExecutor.

    Returns:
        None
    """
    # Clean patchlets directory
    LOGGER.info(f'Cleaning patchlets directory {PATCHLETS_DIR}')
    PATCHLETS_DIR.mkdir(parents=True, exist_ok=True)
    for item in PATCHLETS_DIR.iterdir():
        if item.is_dir() and item.name.startswith('patchlet_'):
            shutil.rmtree(item)
        elif item.is_file() and item.name.startswith('patchlet_'):
            item.unlink()

    # Get all EOPatches
    eopatches = [eopatch_path.absolute() for eopatch_path in EOPATCH_DIR.iterdir()
                 if eopatch_path.is_dir()]
    LOGGER.info(f'Creating patchlets for {len(eopatches)} EOPatches')

    # Create patchlets in parallel
    total_patchlets_created = 0
    total_patchlets_best_case = len(eopatches) * EOTASK_NUM_SAMPLES
    with ProcessPoolExecutor() as executor:
        futures = []
        with tqdm(total=len(eopatches)) as pbar:
            for eopatch_path in eopatches:
                future = executor.submit(create_patchlets, eopatch_path)
                futures.append(future)
            for future in as_completed(futures):
                try:
                    result = future.result()
                    total_patchlets_created += result
                except Exception as e:
                    LOGGER.error(f'A task failed: {e}')
                pbar.update(1)

    LOGGER.info(
        f'Created {total_patchlets_created} patchlets out of {total_patchlets_best_case}'
        f' Or {total_patchlets_created/total_patchlets_best_case*100:.2f}% of the best case')
    return total_patchlets_created


if __name__ == "__main__":
    if NIVA_PROJECT_DATA_ROOT is None:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable should be set.")
    create_all_patchlets()

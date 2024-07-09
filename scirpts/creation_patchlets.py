import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Tuple
from eolearn.core import EOPatch, FeatureType, FeatureTypeSet, EOTask
from sentinelhub import BBox
import fs.move
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class SamplePatchlets(EOTask):
    INVALID_DATA_FRACTION = 0.0
    S2_RESOLUTION = 10

    def __init__(self, feature: Tuple[FeatureType, str], buffer: int, patch_size: int, num_samples: int,
                 max_retries: int, sample_features: List[Tuple[FeatureType, str]], fraction_valid: float = 0.2,
                 no_data_value: int = 0, sample_positive: bool = True):
        self.feature_type, self.feature_name, self.new_feature_name = next(
            self._parse_features(feature, new_names=True,
                                 default_feature_type=FeatureType.MASK_TIMELESS,
                                 allowed_feature_types={FeatureType.MASK_TIMELESS},
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
        return ratio < self.fraction if self.sample_positive else ratio > self.fraction

    def execute(self, eopatch: EOPatch, seed: int = None) -> List[EOPatch]:
        timestamps = np.array(eopatch.timestamp)
        mask = eopatch[self.feature_type][self.feature_name].squeeze()

        # Convert mask to int32 to avoid DeprecationWarning
        if mask.dtype != np.int32 and mask.dtype != bool:
            mask = mask.astype(np.int32)

        # Ensure mask is 3-dimensional
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]

        n_rows, n_cols, _ = mask.shape

        if mask.ndim != 3:
            raise ValueError('Invalid shape of sampling reference map.')

        # Check if the patch size and buffer are valid
        if self.patch_size > n_rows or self.patch_size > n_cols:
            raise ValueError(f"Patch size {self.patch_size} is too large for the mask dimensions {n_rows}x{n_cols}.")

        np.random.seed(seed)
        eops_out = []

        for patchlet_num in range(0, self.num_samples):
            ratio = 0.0 if self.sample_positive else 1
            retry_count = 0
            new_eopatch = EOPatch(timestamp=eopatch.timestamp)
            
            while self._area_fraction_condition(ratio) and retry_count < self.max_retries:
                if n_rows - self.patch_size - self.buffer <= 0 or n_cols - self.patch_size - self.buffer <= 0:
                    LOGGER.warning(f"The patch size {self.patch_size} and buffer {self.buffer} are too large for the mask dimensions {n_rows}x{n_cols}. Trying without buffer.")
                    row = 0
                    col = 0
                else:
                    row = np.random.randint(self.buffer, n_rows - self.patch_size - self.buffer)
                    col = np.random.randint(self.buffer, n_cols - self.patch_size - self.buffer)
                patchlet = mask[row:row + self.patch_size, col:col + self.patch_size]
                ratio = np.sum(patchlet != self.no_data_value) / self.patch_size ** 2
                retry_count += 1

            if retry_count == self.max_retries:
                LOGGER.warning(f'Could not determine an area with good enough ratio of valid sampled pixels for '
                               f'patchlet number: {patchlet_num}')
                continue

            for feature_type, feature_name in self.sample_features:
                if feature_type in FeatureTypeSet.RASTER_TYPES.intersection(FeatureTypeSet.SPATIAL_TYPES):
                    feature_data = eopatch[feature_type][feature_name]
                    if feature_type.is_time_dependent():
                        sampled_data = feature_data[:, row:row + self.patch_size, col:col + self.patch_size, :]
                    else:
                        sampled_data = feature_data[row:row + self.patch_size, col:col + self.patch_size, :]

                    # Add the sampled data to the new EOPatch
                    new_eopatch.add_feature(feature_type, feature_name, sampled_data)

            eops_out.append(new_eopatch)

        return eops_out

if __name__ == "__main__":
    # Configuration for sampling patchlets
    eopatch_path = '/home/joseph/Code/localstorage/eopatchs/eopatch_20240704_092728'
    output_path = '/home/joseph/Code/localstorage/patchlets'
    sample_positive = True
    mask_feature_name = 'EXTENT'
    buffer = 0
    patch_size = 256
    num_samples = 10
    max_retries = 10
    fraction_valid = 0.4
    sampled_feature_name = 'BANDS'

    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)

    task = SamplePatchlets(
        feature=(FeatureType.MASK_TIMELESS, mask_feature_name),
        buffer=buffer,
        patch_size=patch_size,
        num_samples=num_samples,
        max_retries=max_retries,
        fraction_valid=fraction_valid,
        sample_features=[
            (FeatureType.DATA, sampled_feature_name),
            (FeatureType.MASK_TIMELESS, 'EXTENT'),
            (FeatureType.MASK_TIMELESS, 'BOUNDARY'),
            (FeatureType.MASK_TIMELESS, 'DISTANCE')
        ],
        sample_positive=sample_positive
    )

    patchlets = task.execute(eopatch)

    for i, patchlet in enumerate(patchlets):
        patchlet_path = os.path.join(output_path, f'patchlet_{i}')
        if os.path.exists(patchlet_path):
            shutil.rmtree(patchlet_path)
        patchlet.save(patchlet_path)
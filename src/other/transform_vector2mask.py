import os
import sys
from typing import Tuple
import geopandas as gpd
from dataclasses import dataclass
import numpy as np
from skimage.morphology import binary_dilation, disk
from shapely.geometry import box

import fs.move  # required by eopatch.save
from eolearn.core import FeatureType, EOPatch, OverwritePermission
from eolearn.geometry import VectorToRaster

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)


@dataclass
class GsaaToEopatchConfig:
    vector_file_path: str
    feature_name: str
    vector_feature: Tuple[FeatureType, str]
    extent_feature: Tuple[FeatureType, str]
    boundary_feature: Tuple[FeatureType, str]
    #distance_feature: Tuple[FeatureType, str]
    eopatches_folder: str
    buffer_poly: int = -10
    no_data_value: int = 0
    width: int = 1000
    height: int = 1000
    disk_radius: int = 2


def create_eopatch(vector_file_path,
                   eopatch_path=None,
                   feature_name='CADASTRE'):
    """
    vector_data: GeoPandas
    """
    LOGGER.info(f"load EOPatch from {eopatch_path}")
    eopatch = EOPatch.load(eopatch_path)
    croped_bounds = gpd.GeoSeries([box(*eopatch.bbox)], crs=eopatch.bbox.crs.epsg)
    vector_data = gpd.read_file(vector_file_path, bbox=croped_bounds)
    # if not the same crs
    vector_data = vector_data.to_crs(epsg=eopatch.bbox.crs.epsg)

    eopatch.add_feature(FeatureType.VECTOR_TIMELESS, feature_name, vector_data)
    # Create and save unique EOPatch
    eopatch.save(eopatch_path,
                 overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    # LOGGER.info(eopatch)
    return eopatch


def main_rastorize_vector(config):
    # get extent mask from vector
    config = GsaaToEopatchConfig(
        vector_file_path=config['vector_file_path'],
        eopatches_folder=config['eopatches_folder'],
        feature_name=config['feature_name'],
        vector_feature=(FeatureType(config['vector_feature'][0]), config['vector_feature'][1]),
        extent_feature=(FeatureType(config['extent_feature'][0]), config['extent_feature'][1]),
        boundary_feature=(FeatureType(config['boundary_feature'][0]), config['boundary_feature'][1]),
        # distance_feature=(FeatureType(config['distance_feature'][0]), config['distance_feature'][1]),
        # height=rasterize_config['height'],
        # width=rasterize_config['width'],  # shapes not sure
    )

    eopatch = create_eopatch(vector_file_path=config.vector_file_path,
                             eopatch_path=config.eopatches_folder,
                             feature_name=config.feature_name)

    # VectorToRaster
    # https://github.com/sentinel-hub/eo-learn/blob/df8bbe80a0a0dbd9326c05b2c2d94ff41b152e3d/eolearn/geometry/transformations.py#L42
    # raster_resolution ? not raster_shape
    vec2ras = VectorToRaster(config.vector_feature,
                             config.extent_feature,
                             values=1,  #raster_shape=(config.width, config.height),
                             raster_resolution=(10, 10), # right way
                             no_data_value=config.no_data_value,
                             buffer=config.buffer_poly, write_to_existing=False)
    vec2ras.execute(eopatch)
    # add boundary
    eopatch = extent2boundary(eopatch, extent_feature=config.extent_feature[1],
                              boundary_feature=config.boundary_feature[1])
    eopatch.save(config.eopatches_folder,
                 overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    # LOGGER.info(eopatch)
    return eopatch


def extent2boundary(eopatch, extent_feature, boundary_feature, structure=disk(2)):
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/gsaa_to_eopatch.py#L87
    extent_mask = eopatch.mask_timeless[extent_feature].squeeze(axis=-1)
    boundary_mask = binary_dilation(extent_mask, selem=structure) - extent_mask
    eopatch.add_feature(FeatureType.MASK_TIMELESS, boundary_feature,
                        boundary_mask[..., np.newaxis])
    return eopatch

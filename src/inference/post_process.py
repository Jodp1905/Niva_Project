import sys
import os
import geopandas as gpd

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Project imports
from training.tf_data_utils import normalize_meanstd  # noqa: E402

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)

# Load configuration
from niva_utils.config_loader import load_config  # noqa: E402
CONFIG = load_config()

# Constants
NIVA_PROJECT_DATA_ROOT = CONFIG['niva_project_data_root']
POSTPROCESS_CONFIG = CONFIG['postprocess_config']

# Inferred constants
GPKG_PATH = os.path.join(NIVA_PROJECT_DATA_ROOT,
                         "inference", "contours", "merged_0_32631.gpkg")


def final_post_process(config):
    # Post-process (simplify, filter small polygons, multipolygon ?, save to geojson)
    path = config["gpkg_path"]
    LOGGER.info(f'Final post-processing for {path} file')
    vectors = gpd.read_file(path)
    new_vectors = vectors.geometry.simplify(
        tolerance=config["simplify_tolerance"], preserve_topology=True)
    # filter small, big polygons
    new_vectors = new_vectors[new_vectors.geometry.area >
                              config["smallest_area"]]
    new_vectors = new_vectors[new_vectors.geometry.area <
                              config["biggest_area"]]

    folder, file = os.path.split(path)
    os.makedirs(os.path.join(folder, f'v_{config["version"]}'), exist_ok=True)
    # crs to lon, lat
    new_vectors = new_vectors.to_crs("epsg:4326")
    path_json = os.path.join(folder, f'v_{config["version"]}',
                                     f'{os.path.splitext(file)[0]}.geojson')
    new_vectors.to_file(path_json,
                        driver='GeoJSON')
    LOGGER.info(f"Saved GeoJson file to {path_json}")
    return new_vectors


def main_postprocess():
    POSTPROCESS_CONFIG["gpkg_path"] = GPKG_PATH
    final_post_process(POSTPROCESS_CONFIG)
    LOGGER.info("Post-processing finished")

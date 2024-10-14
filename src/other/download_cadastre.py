import logging
import os
import sys
import geopandas as gpd
import numpy as np
import requests
from tqdm import tqdm
import py7zr  # for extraction

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
CADASTRE_CONFIG = CONFIG['download_cadastre']
# Constants
PROJECT_DATA_ROOT = CONFIG['niva_project_data_root_inf']
TILE_ID = CONFIG['TILE_ID']
FLAG_DOWNLOAD = CADASTRE_CONFIG["flag_download"]  # download from url or file already exists
sim_tolerance = CADASTRE_CONFIG["sim_tolerance"]
chunk_size = CADASTRE_CONFIG["chunk_size"]
# Inferred constants
PROJECT_DATA_ROOT = os.path.join(PROJECT_DATA_ROOT, TILE_ID)
# The ground truth data from https://geoservices.ign.fr/rpg
# RPG_2-2__SHP_LAMB93_R27_2023-01-01 (région Bourgogne-Franche-Comté)
# "RPG_2-2__SHP_LAMB93_R53_2023-01-01"  # région Bretagne
sub_name = CADASTRE_CONFIG["sub_name"]
url_brittany = f"https://data.geopf.fr/telechargement/download/RPG/{sub_name}/{sub_name}.7z"
# or ILOTS_DE_REFERENCE isolated plots ?
file_name = "PARCELLES_GRAPHIQUES"
archive_sub_path = f"{sub_name}\\RPG\\1_DONNEES_LIVRAISON_2023\\{sub_name}/{file_name}.shp"

TILE_META = os.path.join(PROJECT_DATA_ROOT, "tile", f"metadata.gpkg")  # metadata of the tile (input)
# will create during the run
CADASTRE_FINAL_PATH = os.path.join(PROJECT_DATA_ROOT,
                                   f"CAP.7z")
CADASTRE_FINAL_TILE_PATH = os.path.join(PROJECT_DATA_ROOT, "tile",
                                        f"{file_name}_{TILE_ID[4:9]}.gpkg")
SIMPLIFIED_CADASTRE_PATH = os.path.join(PROJECT_DATA_ROOT, "tile",
                                        f"{file_name}_{TILE_ID[4:9]}_simplified_t_{sim_tolerance}.gpkg")


def main():
    if FLAG_DOWNLOAD:  # curl, wget
        with requests.get(url_brittany, stream=True) as response:
            with open(CADASTRE_FINAL_PATH, mode="wb") as file:
                for chunk in tqdm(response.iter_content(chunk_size=chunk_size),
                                  desc="downloading file"):
                    file.write(chunk)

        # multi volume
        # with multivolumefile.open('example.7z', mode='rb') as target_archive:
        #     with SevenZipFile(target_archive, 'r') as archive:
        #         archive.extractall()
        # python -m py7zr x monty.7z target-dir/
        with py7zr.SevenZipFile(CADASTRE_FINAL_PATH, mode='r') as z:
            z.extractall(path=PROJECT_DATA_ROOT)

    tile_meta = gpd.read_file(TILE_META)
    # tile_bounds = tile_meta.total_bounds
    tile_gt_path = os.path.join(PROJECT_DATA_ROOT, archive_sub_path)
    # read only the data for the given tile bounds
    data_gt = gpd.read_file(tile_gt_path, bbox=tile_meta.geometry)  # [386588 rows x 7 columns]
    LOGGER.info(f"saving to {CADASTRE_FINAL_TILE_PATH}")
    data_gt.to_file(CADASTRE_FINAL_TILE_PATH, index=False)

    """Simple data analysis below"""

    # Columns: [ID_PARCEL, SURF_PARC, CODE_CULTU, CODE_GROUP, CULTURE_D1, CULTURE_D2, geometry]
    LOGGER.info(f"file {tile_gt_path} len {len(data_gt)} and columns {data_gt.columns}")
    LOGGER.info(f"file {tile_gt_path} crs {data_gt.crs}")
    # no MultiPolygons
    LOGGER.info(f"other geometries except Polygon in dataset = {data_gt[data_gt.geometry.type != 'Polygon']}")

    coor_counts = data_gt.geometry.count_coordinates()
    rings_count = data_gt.geometry.count_interior_rings()
    LOGGER.info(f"Interior ring count min = {np.min(rings_count)} max = {np.max(rings_count)}")  # 0, 35
    LOGGER.info(f"Number of geometries with interior ring = {len(data_gt[rings_count != 0])}")  # 6015
    LOGGER.info(f"Number of coordinates for geometries min = {np.min(coor_counts)}, "
                f"max = {np.max(coor_counts)}, mean = {np.mean(coor_counts)}")  # 4 836 23.5

    data_gt_sim = data_gt.geometry.simplify(tolerance=sim_tolerance,
                                            preserve_topology=True)
    coor_counts = data_gt_sim.geometry.count_coordinates()
    LOGGER.info(f"Number of coordinates for geometries after simplification min = {np.min(coor_counts)}, "
                f"max = {np.max(coor_counts)}, mean = {np.mean(coor_counts)}")  # 4 352 9.4
    data_gt_sim.to_file(SIMPLIFIED_CADASTRE_PATH)


if __name__ == "__main__":
    main()

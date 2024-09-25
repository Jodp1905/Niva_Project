from pystac_client import Client
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import pystac_client
import dask.distributed
import folium
import folium.plugins
import geopandas as gpd
import shapely.geometry
import yaml
from branca.element import Figure
from IPython.display import HTML, display
import odc.ui
from odc.stac import stac_load
from geojson.utils import coords
from shapely.geometry import LineString
from shapely.geometry import shape
import os
import sys
import geojson
from pathlib import Path
import xarray as xr
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
# INPUT_COORDS_PATH = CONFIG['inference']['input_coords_path']
INPUT_COORDS_PATH = "/home/jules/stage/Niva_Project/src/inference/input_coords.geojson"
TILE_PATH = Path(f'{NIVA_PROJECT_DATA_ROOT}/tile/inference_tile.nc')


def open_input_coords_file(geojson_file_path):
    if not os.path.exists(geojson_file_path):
        raise FileNotFoundError(f"File {geojson_file_path} does not exist")
    with open(geojson_file_path) as f:
        geojson_data = geojson.load(f)
        return geojson_data


def get_bbox(geometry):
    return LineString(coords(geometry)).bounds


def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation

    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))


def get_search_json(datatime, bbox,
                    filters=None,
                    url_l="https://earth-search.aws.element84.com/v1",
                    collections=["sentinel-2-l2a"]):
    catalog = Client.open(url_l)
    query = catalog.search(
        collections=collections, datetime=datatime, limit=100,
        query=filters,
        bbox=bbox,
        # sortby=[
        #   {
        #       "field": "properties.datatime",
        #       "direction": "asc"
        #   },]
    )
    items = list(query.items())
    if not items:
        return None, None
    # Convert STAC items into a GeoJSON FeatureCollection
    stac_json = query.item_collection_as_dict()
    return stac_json, items


def get_gdf(stac_json):
    gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")

    # Compute granule id from components
    gdf["granule"] = (
        gdf["mgrs:utm_zone"].apply(lambda x: f"{x:02d}")
        + gdf["mgrs:latitude_band"]
        + gdf["mgrs:grid_square"]
    )
    return gdf


def plot_bounds(gdf):
    fig = gdf.plot(
        "granule",
        edgecolor="black",
        categorical=True,
        aspect="equal",
        alpha=0.5,
        figsize=(6, 12),
        legend=True,
        legend_kwds={"loc": "upper left", "frameon": False, "ncol": 1},
    )
    _ = fig.set_title("STAC Query Results")


def plot_bounds_map(gdf, bbox, name_g="STAC", tooltip=[
    "granule",
    "datetime",
    "s2:nodata_pixel_percentage",
    "eo:cloud_cover",
    "s2:vegetation_percentage",
    "s2:water_percentage",
]):
    fig = Figure(width="400px", height="500px")
    map1 = folium.Map()
    fig.add_child(map1)

    folium.GeoJson(
        shapely.geometry.box(*bbox),
        style_function=lambda x: dict(
            fill=False, weight=2, opacity=0.7, color="olive"),
        name="Query",
        tooltip="Query",
    ).add_to(map1)

    gdf.explore(
        "granule",
        categorical=True,
        tooltip=tooltip,
        popup=True,
        style_kwds=dict(fillOpacity=0.1, width=2),
        name=name_g,
        m=map1,
    )

    map1.fit_bounds(bounds=convert_bounds(gdf.unary_union.bounds))
    display(fig)


def download_tile():
    if TILE_PATH is None:
        raise ValueError("Tile path is not defined, please set it in the configuration YAML file"
                         " or with the environment variable 'TILE_PATH'")
    datatime = "2023-02-01/2023-04-30"
    filters = {
        "eo:cloud_cover": {"lt": 0.6},
        # s2:nodata_pixel_percentage
        "mgrs:grid_square": {"eq": "EN"}
    }
    bbox_json = open_input_coords_file(INPUT_COORDS_PATH)
    bbox = get_bbox(bbox_json)
    LOGGER.info(f"Searching for data in {bbox}")
    stac_json, items = get_search_json(datatime, bbox, filters)
    if stac_json is None:
        LOGGER.info("No items found")
        return
    LOGGER.info(f"Found {len(items)} items")
    if items:
        gdf = get_gdf(stac_json)
        plot_bounds(gdf)
        plot_bounds_map(gdf, bbox)
    # ["B02", "B03", "B04", "B08", "SCL"]
    bands = ["blue", "green", "red", "nir", "scl"]
    # downloading the tile specified bands
    tidx = 0  # only one timestamp / acquisition
    ds: xr.Dataset = odc.stac.stac_load([items[tidx]], bands=bands,
                                        chunks={}, resolution=10)
    ds = ds.rename_vars(
        {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8"})
    LOGGER.info(f"Downloading tile: {items[tidx].id} under {TILE_PATH}")
    time_before = time()
    ds.load().to_netcdf(TILE_PATH)
    time_after = time()
    LOGGER.info(f"Downloaded tile in {time_after - time_before} seconds"
                f" to {TILE_PATH}")


if __name__ == "__main__":
    download_tile()

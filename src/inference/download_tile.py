import geojson.feature
from pystac_client import Client as STACClient
import geopandas as gpd
import xarray as xr
import os
import sys
from pathlib import Path
from shapely.geometry import LineString
import geojson
from time import time
from geojson.utils import coords
import odc.stac
import folium
import shapely
from branca.element import Figure
from dask.diagnostics import ProgressBar
from dask import config as dask_config
import http.server
import socketserver
import webbrowser
import threading
import tqdm
from typing import List, Tuple

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
INPUT_COORDS_PATH = CONFIG['download_tile']['input_coords_path']
XARRAY_CHUNK_SIZE = CONFIG['download_tile']['xarray_chunk_size']

# Inferred constants
TILE_FOLDER = Path(f"{NIVA_PROJECT_DATA_ROOT}/inference/tile")


def serve_map(tile_path: str,
              port: int = 8887) -> None:
    """
    Serves a map tile using a simple HTTP server.

    Args:
        tile_path (str): The file path to the map tile to be served.
        port (int, optional): The port number on which the server will listen. Defaults to 8887.

    Side Effects:
        Changes the current working directory to the directory containing the tile file.
        Opens the default web browser to the URL where the map tile is being served.
        Starts an HTTP server that serves the map tile indefinitely.

    Example:
        serve_map("/path/to/tile.html", port=8080)
    """
    directory = os.path.dirname(os.path.abspath(tile_path))
    handler = http.server.SimpleHTTPRequestHandler
    os.chdir(directory)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(
            f"Serving at http://localhost:{port}/{os.path.basename(tile_path)}")
        webbrowser.open(
            f"http://localhost:{port}/{os.path.basename(tile_path)}")
        httpd.serve_forever()


def open_input_coords_file(geojson_file_path: str) -> dict:
    """
    Opens a GeoJSON file and returns its contents.

    Args:
        geojson_file_path (str): The file path to the GeoJSON file.

    Returns:
        dict: The contents of the GeoJSON file as a dictionary.

    Raises:
        FileNotFoundError: If the specified GeoJSON file does not exist.
    """
    if not os.path.exists(geojson_file_path):
        raise FileNotFoundError(f"File {geojson_file_path} does not exist")
    with open(geojson_file_path) as f:
        geojson_data = geojson.load(f)
        return geojson_data


def get_bbox(geometry: geojson.feature.FeatureCollection) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box of a given geometry.

    Args:
        geometry (shapely.geometry.base.BaseGeometry): 
        The input geometry for which the bounding box is to be calculated.

    Returns:
        tuple: A tuple representing the bounding box in the format (minx, miny, maxx, maxy).
    """
    res = LineString(coords(geometry)).bounds
    return res


def convert_bounds(bbox: Tuple[float, float, float, float],
                   invert_y: bool = False) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Converts bounding box representation to leaflet notation (y, x).

    Args:
        bbox (tuple): A tuple of four elements representing the bounding box (x1, y1, x2, y2).
        invert_y (bool, optional): If True, inverts the y-coordinates. Defaults to False.

    Returns:
        tuple: A tuple of tuples representing the corners ((y1, x1), (y2, x2)).
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))


def get_search_json(datatime: str,
                    bbox: Tuple[float, float, float, float],
                    filters: dict = None,
                    url_l: str = "https://earth-search.aws.element84.com/v1",
                    collections: List[str] = ["sentinel-2-l2a"]) -> Tuple[dict, List]:
    """
    Searches for satellite imagery data using the STAC API and returns the results in GeoJSON format.

    Args:
        datatime (str): The datetime range for the search in ISO 8601 format.
        bbox (list): The bounding box for the search as [min_lon, min_lat, max_lon, max_lat].
        filters (dict, optional): Additional filters for the search query. 
        Defaults to None.
        url_l (str, optional): The base URL for the STAC API. 
        Defaults to "https://earth-search.aws.element84.com/v1".
        collections (list, optional): The collections to search within. 
        Defaults to ["sentinel-2-l2a"].

    Returns:
        tuple: A tuple containing:
            - stac_json (dict or None): The search results as a GeoJSON FeatureCollection, 
            or None if no items found.
            - items (list or None): A list of STAC items, or None if no items found.
    """
    catalog = STACClient.open(url_l)
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
    """
    Converts a STAC JSON object to a GeoDataFrame and computes the granule ID.

    Args:
        stac_json (dict): A dictionary representing the STAC JSON object.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the geometries and attributes 
        from the STAC JSON, with an additional 'granule' column representing the granule ID.
    """
    gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")
    # Compute granule id from components
    gdf["granule"] = (
        gdf["mgrs:utm_zone"].apply(lambda x: f"{x:02d}")
        + gdf["mgrs:latitude_band"]
        + gdf["mgrs:grid_square"]
    )
    return gdf


def plot_bounds_map(gdf: gpd.GeoDataFrame,
                    bbox: Tuple[float, float, float, float],
                    name_g: str = "STAC",
                    tooltip: List[str] = [
                        "granule",
                        "datetime",
                        "s2:nodata_pixel_percentage",
                        "eo:cloud_cover",
                        "s2:vegetation_percentage",
                        "s2:water_percentage"],
                    create_live_server: bool = False) -> None:
    """
    Plots a map with the given GeoDataFrame and bounding box.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing the geometries to plot.
    bbox (tuple): A tuple representing the bounding box (minx, miny, maxx, maxy).
    name_g (str, optional): The name for the GeoDataFrame layer. Defaults to "STAC".
    tooltip (list, optional): A list of column names to include in the tooltip.
    create_live_server (bool, optional): If True, creates a live server to serve the map. 
    Defaults to False.

    Returns:
    None
    """
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
    hmtl_map_path = os.path.join(TILE_FOLDER, "map.html")
    map1.save(hmtl_map_path)
    LOGGER.info(f"Map saved to {hmtl_map_path}")
    if create_live_server:
        threading.Thread(target=serve_map, args=(
            hmtl_map_path,), daemon=True).start()


def download_tile(use_dask: bool = True):
    """
    Downloads a tile of satellite imagery data based on specified parameters and saves it to disk.

    Parameters:
    use_dask (bool): If True, use Dask for parallel computation. Defaults to True.

    Raises:
    ValueError: If the path for the input coordinate GeoJSON file is not defined.

    Returns:
    None
    """
    if INPUT_COORDS_PATH is None:
        raise ValueError("Path for the input coordinate gejson file is not defined, please "
                         "set it in the configuration YAML file "
                         "or with the environment variable 'INPUT_COORDS_PATH'")
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
    LOGGER.info(f"Found {len(items)} items: {items}")
    if items:
        gdf = get_gdf(stac_json)
        plot_bounds_map(gdf, bbox)
    tidx = 0  # Only one timestamp / acquisition
    LOGGER.info(f"Loading tile: {items[tidx].id}")
    # ["B02", "B03", "B04", "B08", "SCL"]
    bands = ["blue", "green", "red", "nir", "scl"]
    item = items[tidx]
    if use_dask:
        ds: xr.Dataset = odc.stac.load(items=[item],
                                       bands=bands,
                                       chunks={'x': XARRAY_CHUNK_SIZE,
                                               'y': XARRAY_CHUNK_SIZE},
                                       resolution=10)
    else:
        ds: xr.Dataset = odc.stac.load(items=[item],
                                       bands=bands,
                                       resolution=10,
                                       progress=tqdm.tqdm)
    # Rename variables
    ds = ds.rename_vars(
        {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8"})
    LOGGER.info(f"Loaded dataset:\n{ds}")
    # Set encoding for compression and optimized I/O performance with Dask
    # https://docs.xarray.dev/en/latest/user-guide/dask.html
    if use_dask:
        encoding = {
            'B2': {'zlib': True, 'complevel': 4,
                   'chunksizes': (1, XARRAY_CHUNK_SIZE, XARRAY_CHUNK_SIZE)},
            'B3': {'zlib': True, 'complevel': 4,
                   'chunksizes': (1, XARRAY_CHUNK_SIZE, XARRAY_CHUNK_SIZE)},
            'B4': {'zlib': True, 'complevel': 4,
                   'chunksizes': (1, XARRAY_CHUNK_SIZE, XARRAY_CHUNK_SIZE)},
            'B8': {'zlib': True, 'complevel': 4,
                   'chunksizes': (1, XARRAY_CHUNK_SIZE, XARRAY_CHUNK_SIZE)},
            'scl': {'zlib': True, 'complevel': 4,
                    'chunksizes': (1, XARRAY_CHUNK_SIZE, XARRAY_CHUNK_SIZE)}
        }
    else:
        encoding = {
            'B2': {'zlib': True, 'complevel': 4},
            'B3': {'zlib': True, 'complevel': 4},
            'B4': {'zlib': True, 'complevel': 4},
            'B8': {'zlib': True, 'complevel': 4},
            'scl': {'zlib': True, 'complevel': 4}
        }
    # Save the tile to disk
    time_before = time()
    filename = "input_tile.nc"
    tile_file = os.path.join(TILE_FOLDER, filename)
    if os.path.exists(tile_file):
        LOGGER.info(f"Detecting existing tile file {tile_file}, removing it")
        os.remove(tile_file)
    delayed_obj = ds.to_netcdf(path=tile_file,
                               encoding=encoding,
                               compute=False)
    LOGGER.debug("Delayed object created")
    if use_dask:
        with dask_config.set(scheduler='threads'):
            with ProgressBar():
                delayed_obj.compute()
    else:
        delayed_obj.compute()
    time_after = time()
    f"Saved tile in {time_after - time_before} seconds to {tile_file}"


if __name__ == "__main__":
    TILE_FOLDER.mkdir(parents=True, exist_ok=True)
    download_tile()

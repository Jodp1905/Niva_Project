import warnings
import sys
import logging
import os
import json
from datetime import datetime, timezone
from numpy import newaxis, concatenate
from concurrent.futures import ProcessPoolExecutor, as_completed
from xarray import open_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.filters import rank
from skimage.morphology import disk
from glob import glob
import numpy as np
from numpy import datetime_as_string
import geopandas as gpd
from xarray import DataArray

import tensorflow as tf

# TODO delete dependency on sentinelhub
from sentinelhub.geometry import BBox # https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.geometry.html
from sentinelhub.geometry import CRS # https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.constants.html#sentinelhub.constants.CRS
import fs.move  # required by eopatch.save
from eolearn.io import ExportToTiffTask
from eolearn.core import EOPatch, FeatureType
from eolearn.core import OverwritePermission
from eoflow.models.segmentation_unets import ResUnetA

from vectorisation import run_vectorisation, VectorisationConfig
from filter import LogFileFilter

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Suppress DeprecationWarning
warnings.filterwarnings("ignore")


def normalize_meanstd(ds_keys: dict, subtract: str = 'mean') -> dict:
    # from training code !!!!!!!! # https://github.com/Jodp1905/Niva_Project/blob/91e56139f2bfcb8e60f63f3d20e2e4950b516384/scripts/tf_data_utils.py#L75
    """
    Help function to normalise the features by the mean and standard deviation.
    This is a normalization function that follows recent state-of-the-art solutions.
    There are two methods, that depends on mean, std = None parameters to method normalize.
    If mean, std = None, than this algorithm will be as in https://github.com/antofuller/CROMA,
    else as in https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
    """
    # code for normalization follows parameters here
    # https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
    S2A_MEAN = {
        'B2': 889.6,
        'B3': 1151.7,
        'B4': 1307.6,
        'B8': 2538.9, }
    S2A_STD = {
        'B2': 1159.1,
        'B3': 1188.1,
        'B4': 1375.2,
        'B8': 1476.4, }
    mean = tf.constant(list(S2A_MEAN.values()), dtype=tf.float64)
    std = tf.constant(list(S2A_STD.values()), dtype=tf.float64)

    def normalize(feats, mean=None, std=None):
        # follows normalization presented https://github.com/zhu-xlab/SSL4EO-S12/tree/main
        feats = tf.cast(feats, tf.float64)
        if mean is None or std is None:
            # normalization method follows presented here https://github.com/antofuller/CROMA
            # axis to reduce, all except last where bands are
            inds = list(range(len(feats.shape) - 1))
            std = tf.math.reduce_std(feats, inds)
            mean = tf.math.reduce_mean(feats, inds)
        std_2 = tf.math.multiply(tf.constant(2.0, dtype=tf.float64), std)
        min_value = tf.math.subtract(mean, std_2)
        max_value = tf.math.add(mean, std_2)
        div = tf.math.subtract(max_value, min_value)
        feats = tf.math.subtract(feats, min_value)
        feats = tf.math.divide(feats, div)
        feats = tf.clip_by_value(feats, clip_value_min=0., clip_value_max=1.)
        return feats
    feats = normalize(ds_keys['features'], mean=mean, std=std)
    ds_keys['features'] = feats
    return ds_keys


def transform_timestamps(time_data: DataArray) -> list:
    # from training code https://github.com/Jodp1905/Niva_Project/blob/91e56139f2bfcb8e60f63f3d20e2e4950b516384/scripts/eopatches_for_sampling.py#L42
    # TODO make do without this function
    """
    Transforms xarray.DataArray timestamps to ISO 8601 strings with utc timezone.

    Args:
        time_data (DataArray): The input xarray.DataArray timestamps.

    Returns:
        list: A list of transformed timestamps as datetime objects with utc timezone.
    """
    time_strings = datetime_as_string(time_data, unit='ms')
    transformed_timestamps = []
    for t in time_strings:
        dt = datetime.strptime(
            t, '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        transformed_timestamps.append(dt)
    return transformed_timestamps


def create_eopatch_ds(ds, output_eopatch_dir, image_id='0'):
    # according to https://github.com/Jodp1905/Niva_Project/blob/main/scripts/eopatches_for_sampling.py#L61
    # Create a new EOPatch
    eopatch = EOPatch()
    # Add time data to EOPatch
    time_data = ds['time'].values
    transformed_timestamps = transform_timestamps(time_data)
    eopatch.timestamp = transformed_timestamps

    # Initialize a dictionary to group bands and other data
    data_dict = {
        'BANDS': [],
        'CLP': None
    }

    for var in ds.data_vars:
        if var != 'spatial_ref':
            data = ds[var].values
            # Eopatch data should have 4 dimensions: time, height, width, channels
            if data.ndim == 2:
                # Add time and channel dimensions
                data = data[newaxis, ..., newaxis]
            elif data.ndim == 3:
                # Add channel dimension
                data = data[..., newaxis]
            else:
                LOGGER.warning(
                    f"Variable {var} has unexpected number"
                    f"of dimensions: {data.ndim}. Skipping.")
                continue

            # Group bands into a single BANDS feature
            if var.startswith('B'):
                data_dict['BANDS'].append(data)
            elif var == 'CLP':
                data_dict['CLP'] = data
            else:
                eopatch.add_feature(FeatureType.DATA, var, data)

    # Stack bands data along the last dimension (channels)
    if data_dict['BANDS']:
        bands_data = concatenate(data_dict['BANDS'], axis=-1)
        eopatch.add_feature(FeatureType.DATA, 'BANDS', bands_data)

    if data_dict['CLP'] is not None:
        eopatch.add_feature(FeatureType.DATA, 'CLP', data_dict['CLP'])

    # Add bbox to EOPatch
    bbox = BBox(ds.rio.bounds(), CRS(ds.rio.crs.to_string()))
    eopatch.bbox = bbox
    # Create and save unique EOPatch
    eopatch_name = f'eopatch_{image_id}'
    eopatch_path = os.path.join(output_eopatch_dir, eopatch_name)
    os.makedirs(eopatch_path, exist_ok=True)
    eopatch.save(eopatch_path,
                 overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return eopatch


def create_eopatch(path_patch, output_eopatch_dir, image_id='0'):
    ds = open_dataset(path_patch, decode_coords="all")  # decode all needed to load crs
    return create_eopatch_ds(ds, output_eopatch_dir, image_id)


def split_save2eopatch(config):
    ds = open_dataset(config["tile_path"], decode_coords="all")  # decode all needed to load crs

    assert CRS(ds.rio.crs.to_string()).pyproj_crs().axis_info[0].unit_name == 'metre', \
        'The resulting CRS should have axis units in metres.'
    # TODO overlap should be somewhere
    # TODO padding from all 4 sides of the tile
    num_split = 0
    for ind_x in range(config["begin_x"], len(ds.x) - config["width"] + 1, config["width"]):
        for ind_y in range(config["begin_y"], len(ds.y) - config["height"] + 1, config["height"]):
            num_split += 1
            patch = ds.isel(x=slice(ind_x, ind_x + config["width"]),
                            y=slice(ind_y, ind_y + config["height"]))

            create_eopatch_ds(patch, config["eopatches_folder"], image_id=f"x_{ind_x}_y_{ind_y}_{num_split}")

            if config["num_split"] != -1 and num_split == config["num_split"]:
                break
        if config["num_split"] != -1 and num_split == config["num_split"]:
            break

def load_model(model_cfg_path, chkpt_folder, input_shape=(256, 256, 4)):
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/prediction.py
    input_shape = dict(features=[None, *input_shape])
    with open(model_cfg_path, 'r') as jfile:
        model_cfg = json.load(jfile)

    # initialise model from config, build, compile and load trained weights
    model = ResUnetA(model_cfg)
    model.build(input_shape)
    model.net.compile()
    model.net.load_weights(f'{chkpt_folder}/model.ckpt')
    return model


# def save_predictions(eop_path, model):
#     eop = EOPatch.load(eop_path)
#     extent, boundary, distance = model.net.predict(normalize_meanstd({"features": eop.data["BANDS"]})['features'])
#     eop.add_feature(FeatureType.DATA, 'EXTENT_PREDICTED', extent[..., [1]])
#     eop.add_feature(FeatureType.DATA, 'BOUNDARY_PREDICTED', boundary[..., [1]])
#     eop.add_feature(FeatureType.DATA, 'DISTANCE_PREDICTED', distance[..., [1]])
#     eop.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
#     return eop


def save_predictions(eop_path, model, pad_buffer=12,
                     crop_buffer=12):
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/prediction.py

    def crop_array(array: np.ndarray, buffer: int) -> np.ndarray:
        """ Crop height and width of a 4D array given a buffer size. Array has shape B x H x W x C """
        assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'
        return array[:, buffer:-buffer:, buffer:-buffer:, :]

    eop = EOPatch.load(eop_path)
    data = eop.data["BANDS"]
    data = np.pad(data, [(0, 0), (pad_buffer, pad_buffer), (pad_buffer, pad_buffer), (0, 0)], mode='edge')

    extent, boundary, distance = model.net.predict(normalize_meanstd({"features": data})['features'])

    extent = crop_array(extent, buffer=crop_buffer)
    boundary = crop_array(boundary, buffer=crop_buffer)
    distance = crop_array(distance, buffer=crop_buffer)

    eop.add_feature(FeatureType.DATA, 'EXTENT_PREDICTED', extent[..., [1]])
    eop.add_feature(FeatureType.DATA, 'BOUNDARY_PREDICTED', boundary[..., [1]])
    eop.add_feature(FeatureType.DATA, 'DISTANCE_PREDICTED', distance[..., [1]])
    eop.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return eop


def visualize_predictions(eop, tidx=1, viz_factor=2.5):
    # tidx select one timestamp
    fig, axs = plt.subplots(figsize=(15, 5), ncols=3, sharey=True)
    axs[0].imshow(viz_factor * eop.data['BANDS'][tidx][..., [2, 1, 0]] / 10000)
    axs[0].set_title('RGB bands')
    axs[1].imshow(eop.data['EXTENT_PREDICTED'][tidx].squeeze(), vmin=0, vmax=1)
    axs[1].set_title('Extent')
    axs[2].imshow(eop.data['BOUNDARY_PREDICTED'][tidx].squeeze(), vmin=0, vmax=1)
    axs[2].set_title('Boundary');
    plt.show()


def visualize_predictions(eop, tidx=1, row_s=0, row_d=256,
                          col_s=0, col_d=256, viz_factor=3.5,
                          alpha=.2):
    # tidx select one timestamp
    # EOPatch
    fig, axs = plt.subplots(figsize=(15, 10), ncols=3, sharey=True)
    axs[0].imshow(viz_factor * eop.data['BANDS'][tidx][row_s:row_d, col_s:col_d, [2, 1, 0]] / 10000)
    axs[0].set_title('RGB bands')
    axs[0].imshow(eop.data['EXTENT_PREDICTED'][tidx].squeeze()[row_s:row_d, col_s:col_d],
                  vmin=0, vmax=1, alpha=alpha)
    axs[0].set_title('Extent')
    axs[1].imshow(viz_factor * eop.data['BANDS'][tidx][row_s:row_d, col_s:col_d, [2, 1, 0]] / 10000)
    axs[1].set_title('RGB bands')
    axs[1].imshow(eop.data['BOUNDARY_PREDICTED'][tidx].squeeze()[row_s:row_d, col_s:col_d],
                  vmin=0, vmax=1, alpha=alpha)
    axs[1].set_title('Boundary')
    axs[2].imshow(viz_factor * eop.data['BANDS'][tidx][row_s:row_d, col_s:col_d, [2, 1, 0]] / 10000)
    axs[2].set_title('RGB bands')


def upscale_and_rescale(array: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    # taken from https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py
    """ Upscale a given array by a given scale factor using bicubic interpolation """
    assert array.ndim == 2

    height, width = array.shape

    rescaled = np.array(Image.fromarray(array).resize((width * scale_factor, height * scale_factor), Image.BICUBIC))
    rescaled = (rescaled - np.min(rescaled)) / (np.max(rescaled) - np.min(rescaled))

    assert np.sum(~np.isfinite(rescaled)) == 0

    return rescaled


def smooth(array: np.ndarray, disk_size: int = 2) -> np.ndarray:
    # taken from https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py
    """ Blur input array using a disk element of a given disk size """
    assert array.ndim == 2

    smoothed = rank.mean(array, selem=disk(disk_size).astype(np.float32)).astype(np.float32)
    smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))

    assert np.sum(~np.isfinite(smoothed)) == 0

    return smoothed


def compile_up_sample(eopatch, tidx=0, disk_size=2, scale_factor=2):
    # taken from https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py
    """ Run merging and upscaling """
    extent = eopatch.data["EXTENT_PREDICTED"][tidx, ..., 0]
    boundary = eopatch.data["BOUNDARY_PREDICTED"][tidx, ..., 0]

    combined = np.clip(1 + extent - boundary, 0, 2)

    combined = upscale_and_rescale(combined, scale_factor=scale_factor)
    combined = smooth(combined, disk_size=disk_size)
    combined = upscale_and_rescale(combined, scale_factor=scale_factor)
    array = smooth(combined, disk_size=disk_size * scale_factor)

    eopatch.add_feature(FeatureType.DATA_TIMELESS, f'COMBINED_PREDICTED', np.expand_dims(array, axis=-1))
    return eopatch


def save2tiff(eop_path, tiffs_folder, tidx=0, disk_size=2, scale_factor=2):
    eopatch = EOPatch.load(eop_path)
    eopatch = compile_up_sample(eopatch, tidx=tidx, disk_size=disk_size, scale_factor=scale_factor)
    eopatch.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # tid = eopatch.timestamp[tidx].month
    filename = os.path.join(tiffs_folder, f'{os.path.split(eop_path)[1]}-{eopatch.bbox.crs.epsg}.tiff')
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py#L248
    export_task = ExportToTiffTask(feature=(FeatureType.DATA_TIMELESS, f'COMBINED_PREDICTED'),
                                   folder=".",
                                   compress='DEFLATE', image_dtype=np.float32)
    export_task.execute(eopatch, filename=filename)
    return eopatch


def convert2eopatches(TILE_SPLIT_FOLDER, EOPATCHES_FOLDER, PROCESS_POOL_WORKERS=12):
    tile_split_files = sorted(glob(os.path.join(TILE_SPLIT_FOLDER, "*.nc")))

    LOGGER.info(f'Convert to EOPatch for {len(tile_split_files)} nc files')
    results = []
    with ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS) as executor:
        futures = []
        for file_path in tile_split_files:
            image_id = os.path.splitext(os.path.split(file_path)[1])[0].split("_")[-1]
            future = executor.submit(create_eopatch, path_patch=file_path,
                                     output_eopatch_dir=EOPATCHES_FOLDER, image_id=image_id)
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='Converting to EOPatch'):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                LOGGER.error(f'A task failed: {e}')


def run_predict(prediction_config):
    results = []
    eopatches = [f.path for f in os.scandir(
         prediction_config["eopatches_folder"]) if f.is_dir() and f.name.startswith('eopatch')]
    LOGGER.info(f"EOPatch folders to run prediction: {eopatches}")
    # model of Dynamic Shapes
    assert prediction_config["height"] == prediction_config["width"], "Input height & width must be the same!"
    assert prediction_config["height"] % 32 == 0, "Input size must be div by model's filter 32!"

    model = load_model(prediction_config["model_cfg_path"], prediction_config["model_path"],
                       input_shape=(prediction_config["height"],
                                    prediction_config["width"],
                                    prediction_config["n_channels"]))
    for eopatch_path in tqdm(eopatches):
        results.append(save_predictions(eopatch_path, model, pad_buffer=prediction_config["pad_buffer"],
                     crop_buffer=prediction_config["crop_buffer"]))
    return results


def run_combine(config):

    eopatches = [f.path for f in os.scandir(
        config["eopatches_folder"]) if f.is_dir() and f.name.startswith('eopatch')]

    LOGGER.info(f'Combine extent & boundary and save to tiff for {len(eopatches)} EOPatch files')
    results = []
    # TODO delete time_interval from combine and vectorization
    tiffs_folder = config["tiffs_folder"]
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        futures = []
        for file_path in eopatches:
            future = executor.submit(save2tiff, eop_path=file_path,
                                     tiffs_folder=tiffs_folder, tidx=0,
                                     disk_size=config["disk_size"], scale_factor=config["scale_factor"])
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='Combine and save to tiff predictions'):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                LOGGER.error(f'A task failed: {e}')
    return results


def run_vectorize(config):
    config = VectorisationConfig(
        tiffs_folder=config['tiffs_folder'],
        #time_intervals=config['time_intervals'],
        #utms=config['utms'],
        shape=tuple(config['shape']),
        buffer=tuple(config['buffer']),
        weights_file=config['weights_file'],
        vrt_dir=config['vrt_dir'],
        predictions_dir=config['predictions_dir'],
        contours_dir=config['contours_dir'],
        max_workers=config['max_workers'],
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        threshold=config['threshold'],
        cleanup=config['cleanup'],
        skip_existing=config['skip_existing'],
        rows_merging=config['rows_merging']
    )
    result = run_vectorisation(config=config)
    return result


def final_post_process(config):
    # Post-process (simplify, filter small polygons, multipolygon ?, save to geojson)
    path = config["gpkg_path"]
    LOGGER.info(f'Final post-processing for {path} file')
    vectors = gpd.read_file(path)
    new_vectors = vectors.geometry.simplify(tolerance=config["simplify_tolerance"], preserve_topology=True)
    # filter small, big polygons
    new_vectors = new_vectors[new_vectors.geometry.area > config["smallest_area"]]
    new_vectors = new_vectors[new_vectors.geometry.area < config["biggest_area"]]

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

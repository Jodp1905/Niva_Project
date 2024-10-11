import os
import sys
from typing import Tuple
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from eolearn.core import EOPatch, FeatureType, OverwritePermission
from matplotlib import pyplot as plt
import fs.move  # required by eopatch.save

from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, confusion_matrix, accuracy_score, \
    ConfusionMatrixDisplay

import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

from shapely.geometry import box

from eolearn.core import FeatureType, EOPatch, OverwritePermission
from eolearn.geometry import VectorToRaster
from tqdm import tqdm

from utils_plot import draw_true_color, draw_bbox, draw_vector_timeless, draw_mask
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
ACCURACY_CONFIG = CONFIG['accuracy_computation']
PROJECT_DATA_ROOT = CONFIG['niva_project_data_root_inf']
TILE_ID = CONFIG['TILE_ID']
VERSION = f"v_{ACCURACY_CONFIG['version']}"
VISUALIZE = ACCURACY_CONFIG['visualize']
# Inferred constants
PROJECT_DATA_ROOT = os.path.join(PROJECT_DATA_ROOT, TILE_ID)

CADASTRE_FINAL_TILE_PATH = os.path.join(PROJECT_DATA_ROOT, "tile",
                                        f"PARCELLES_GRAPHIQUES_{TILE_ID[4:9]}.gpkg")
# computed from inference (main_inference.py) vector data
PRED_FILE_PATH = os.path.join(PROJECT_DATA_ROOT, "contours", VERSION, f"{TILE_ID}.gpkg")

EOP_DIR = os.path.join(PROJECT_DATA_ROOT, "eopatches")  # here saved rasterized vectors (predicted, gt - cadastre)
# here metrics are saved
METRICS_PATH = os.path.join(PROJECT_DATA_ROOT, f"metrics_{VERSION}.csv")

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
    LOGGER.info(eopatch)
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


def display_cm(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no field', 'field'])
    # 0 - no field, 1 - field
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def compute_iou_from_cm(cm):
    # compute mean iou
    # iou = true_positives / (true_positives + false_positives + false_negatives)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.nanmean(IoU)


def compute_iou(y_pred, y_true, cm_display=True):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm_display:
        display_cm(cm / len(y_true))
    return cm, compute_iou_from_cm(cm)


def comp_scores(mask_pred, mask_gt):
    mask_gt = mask_gt.flatten()
    mask_pred = mask_pred.flatten()
    score_names = ["accuracy_score", "matthews_corrcoef", "cohen_kappa_score"]
    score_vals = []
    score_vals.append(accuracy_score(mask_gt, mask_pred))
    score_vals.append(matthews_corrcoef(mask_gt, mask_pred))
    score_vals.append(cohen_kappa_score(mask_gt, mask_pred))
    cm, iou = compute_iou(mask_pred, mask_gt, cm_display=VISUALIZE)
    score_names.append("iou")
    score_vals.append(iou)

    cm_normalized = cm / cm.sum()
    score_name_ = ["TN (no field t, no field p)", "FP (no field t, field p)",
                   "FN (field t, no field p)", "TP (field t, field p)"]
    score_names.extend(score_name_)
    score_vals.extend(cm_normalized.flatten().tolist())

    score_names.append("cm")
    score_vals.append(cm)
    return score_names, score_vals


def get_masks_pred_gt(eopatch_folder):
    for feature_name, vector_file_path in zip(["CADASTRE", "PREDICTED"],
                                              [CADASTRE_FINAL_TILE_PATH,
                                               PRED_FILE_PATH]):
        rasterise_gsaa_config = {
            "create_eopatch": feature_name == "CADASTRE",
            "vector_file_path": vector_file_path,
            "eopatches_folder": eopatch_folder,
            "feature_name": feature_name,
            "vector_feature": ["vector_timeless", feature_name],
            "extent_feature": ["mask_timeless", f"EXTENT_{feature_name}"],
            "boundary_feature": ["mask_timeless", f"BOUNDARY_{feature_name}"],
            # "distance_feature": ["data_timeless", "DISTANCE"],
        }
        main_rastorize_vector(rasterise_gsaa_config)


def display_metrics(eopatch_folder):
    eopatch = EOPatch.load(eopatch_folder)
    mask_gt = eopatch.mask_timeless['EXTENT_CADASTRE'].squeeze()
    mask_pred = eopatch.mask_timeless['EXTENT_PREDICTED'].squeeze()
    scores_names, score_vals = comp_scores(mask_pred=mask_pred, mask_gt=mask_gt)
    LOGGER.info(f"Metrics for patch {eopatch_folder} \n {dict(zip(scores_names, score_vals))}")
    return scores_names, score_vals


def visualize_eopatch(eop):
    fig, axis = plt.subplots(figsize=(15, 10), ncols=1, sharey=True)
    eop.vector_timeless['CADASTRE'].plot(ax=axis, color='green')
    eop.vector_timeless['PREDICTED'].plot(ax=axis, color='red')
    plt.show()

    for mask_name in ["EXTENT", "BOUNDARY"]:
        time_idx = 0  # only one tile stemp
        fig, ax = plt.subplots(ncols=4, figsize=(15, 20))
        draw_true_color(ax[0], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[0], eop)
        draw_vector_timeless(ax[0], eop, vector_name='CADASTRE', alpha=.3)

        draw_true_color(ax[1], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[1], eop)
        draw_mask(ax[1], eop, time_idx=None, feature_name=f'{mask_name}_CADASTRE', alpha=.3)

        draw_true_color(ax[2], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[2], eop)
        draw_mask(ax[2], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3)

        draw_true_color(ax[3], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        ax[3].grid()
        plt.show()

    for mask_name in ["EXTENT", "BOUNDARY"]:
        time_idx = 0  # only one tile stemp
        fig, ax = plt.subplots(ncols=4, figsize=(15, 20))
        draw_true_color(ax[0], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[0], eop)
        draw_vector_timeless(ax[0], eop, vector_name='PREDICTED', alpha=.3)

        draw_true_color(ax[1], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[1], eop)
        draw_mask(ax[1], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3)

        draw_true_color(ax[2], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[2], eop)
        draw_mask(ax[2], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3,
                  data_timeless=True)  # visualize predicted masks before vectorization step
        draw_true_color(ax[3], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        ax[3].grid()
        plt.show()


def main():
    # read GT files for region, and read Tile predicted agr
    # convert back to pixels and compare by grid (partial loading) geoJson?
    # https://cadastre.data.gouv.fr/data/etalab-cadastre/2024-07-01/geojson/departements/18/
    # contains in parcels buildings too (needs to filter non crop parcels)
    # Luxembourg 2024 (bad weather during March/April > 50% cloud cover)

    eopatches_path = [f.path for f in os.scandir(
        EOP_DIR) if f.is_dir() and f.name.startswith('eopatch')]

    score_vals_list = []
    cms_list = []
    for eopatch_folder in tqdm(eopatches_path):
        # creates gt and predicted masks from vectors (MOST important method)
        get_masks_pred_gt(eopatch_folder)

        if VISUALIZE:
            eopatch = EOPatch.load(eopatch_folder)
            visualize_eopatch(eopatch)

        scores_names, score_vals = display_metrics(eopatch_folder)
        score_vals, cm = score_vals[:-1], score_vals[-1]
        score_vals_list.append(score_vals)
        cms_list.append(cm)

    cm_all = np.stack(cms_list).sum(axis=0)
    iou_all = compute_iou_from_cm(cm_all)
    cm_all = cm_all.astype(float)
    cm_all /= cm_all.sum().astype(cm_all.dtype)
    display_cm(cm_all)

    # create metrics file
    score_vals_list.append(np.array(score_vals_list).mean(axis=0).tolist())
    metrics_final = pd.DataFrame(columns=["eopatch_folder", "iou_all"] + scores_names[:-1])
    metrics_final["eopatch_folder"] = eopatches_path + ["mean"]
    metrics_final[scores_names[:-1]] = score_vals_list
    metrics_final.loc[metrics_final["eopatch_folder"] == "mean", scores_names[-5:-1]] = cm_all.flatten().tolist()
    metrics_final.loc[metrics_final["eopatch_folder"] == "mean", "iou_all"] = iou_all
    metrics_final.to_csv(METRICS_PATH)

    LOGGER.info(f"confusion matrics for the all eopatches combined {cm_all}")
    LOGGER.info(f"Mean Metrics \n {metrics_final.iloc[-1]}")


if __name__ == "__main__":
    main()

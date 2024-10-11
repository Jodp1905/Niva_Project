import sys
import os
import json
from tqdm import tqdm
import numpy as np
import fs.move  # required by eopatch.save
from eolearn.core import EOPatch, FeatureType
from eolearn.core import OverwritePermission
from eoflow.models.segmentation_unets import ResUnetA
from pathlib import Path
from typing import List, Tuple
import tensorflow as tf
import matplotlib.pyplot as plt

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
PREDICTION_CONFIG = CONFIG['prediction_config']


def visualize_predictions(eop, tidx=0, row_s=0, row_d=256, col_s=0,
                          col_d=256, viz_factor=3.5, alpha=.6):
    """
    Visualizes the predictions for a given EOPatch.

    Args:
        eop (EOPatch): EOPatch containing the data and predictions.
        tidx (int): Index for the time dimension.
        Defaults to 0 since we are working with a single timestamp.
        row_s (int): Starting row index for visualization.
        row_d (int): Ending row index for visualization.
        col_s (int): Starting column index for visualization.
        col_d (int): Ending column index for visualization.
        viz_factor (float): Factor to adjust visualization brightness.
        alpha (float): Transparency factor for overlaying predictions.
    """
    fig, axs = plt.subplots(figsize=(15, 10), ncols=3, sharey=True)

    # RGB bands and predicted extent
    axs[0].imshow(viz_factor * eop.data['BANDS'][tidx]
                  [row_s:row_d, col_s:col_d, [2, 1, 0]] / 10000)
    axs[0].set_title('RGB bands')
    axs[0].imshow(eop.data['EXTENT_PREDICTED'][tidx].squeeze()[row_s:row_d, col_s:col_d],
                  vmin=0, vmax=1, alpha=alpha)
    axs[0].set_title('Extent')

    # RGB bands and predicted boundary
    axs[1].imshow(viz_factor * eop.data['BANDS'][tidx]
                  [row_s:row_d, col_s:col_d, [2, 1, 0]] / 10000)
    axs[1].set_title('RGB bands')
    axs[1].imshow(eop.data['BOUNDARY_PREDICTED'][tidx].squeeze()[row_s:row_d, col_s:col_d],
                  vmin=0, vmax=1, alpha=alpha)
    axs[1].set_title('Boundary')

    # Visualize only RGB bands in the third subplot
    axs[2].imshow(viz_factor * eop.data['BANDS'][tidx]
                  [row_s:row_d, col_s:col_d, [2, 1, 0]] / 10000)
    axs[2].set_title('RGB bands')

    # Save the figure
    # fig_path = os.path.join(INFERENCE_FOLDER, "prediction_random_eop.png")
    # plt.savefig(fig_path)
    # LOGGER.info(
    #     f"Saved prediction visualization for an eopatch at {fig_path}")


def save_predictions(eop_path: str,
                     model: ResUnetA,
                     pad_buffer: int = 12,
                     crop_buffer: int = 12) -> EOPatch:
    """
    Save predictions of a model to an EOPatch.

    Args:
        eop_path (str): Path to the EOPatch to load and save.
        model (ResUnetA): The model used for prediction.
        pad_buffer (int, optional): The padding buffer size to apply to the input data.
        Defaults to 12.
        crop_buffer (int, optional): The cropping buffer size to apply to the predicted data.
        Defaults to 12.

    Returns:
        EOPatch: The EOPatch with added prediction features.

    Raises:
        AssertionError: If the input array to crop_array is not 4D.
    """
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/prediction.py

    def crop_array(array: np.ndarray, buffer: int) -> np.ndarray:
        """ Crop height and width of a 4D array given a buffer size.
        Array has shape B x H x W x C """
        assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'
        return array[:, buffer:-buffer:, buffer:-buffer:, :]

    eop = EOPatch.load(eop_path)
    data: np.ndarray = eop.data["BANDS"]
    data = np.pad(data, [(0, 0), (pad_buffer, pad_buffer),
                         (pad_buffer, pad_buffer), (0, 0)], mode='edge')

    extent, boundary, distance = model.net.predict(
        normalize_meanstd({"features": data})['features'])

    extent = crop_array(extent, buffer=crop_buffer)
    boundary = crop_array(boundary, buffer=crop_buffer)
    distance = crop_array(distance, buffer=crop_buffer)

    eop.add_feature(FeatureType.DATA, 'EXTENT_PREDICTED', extent[..., [1]])
    eop.add_feature(FeatureType.DATA, 'BOUNDARY_PREDICTED', boundary[..., [1]])
    eop.add_feature(FeatureType.DATA, 'DISTANCE_PREDICTED', distance[..., [1]])
    eop.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return eop


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


def run_predict(prediction_config: dict) -> List[EOPatch]:
    """
    Run predictions on a set of EOPatches based on the provided configuration.

    Args:
        prediction_config (dict): A dictionary containing the configuration for predictions.

    Returns:
        List[EOPatch]: A list of EOPatch objects with predictions.
    """
    results = []
    eopatches = [f.path for f in os.scandir(
        prediction_config["eopatches_folder"]) if f.is_dir() and f.name.startswith('eopatch')]
    LOGGER.info(f"Found {len(eopatches)} EOPatches for prediction.")
    # model of Dynamic Shapes
    assert prediction_config["height"] == prediction_config["width"], "Input height & width must be the same!"
    assert prediction_config["height"] % 32 == 0, "Input size must be div by model's filter 32!"

    model = load_model(prediction_config["model_cfg_path"], prediction_config["model_path"],
                       input_shape=(prediction_config["height"],
                                    prediction_config["width"],
                                    prediction_config["n_channels"]))
    for eopatch_path in tqdm(eopatches):
        results.append(save_predictions(eopatch_path,
                                        model,
                                        pad_buffer=prediction_config["pad_buffer"],
                                        crop_buffer=prediction_config["crop_buffer"]))
    return results


def main_prediction(EOPATCHES_FOLDER):
    """
    Main function to perform predictions using a pre-trained model.

    Raises:
        ValueError: If the model path is not defined in the configuration file.
        FileNotFoundError: If the model folder does not exist.
        FileNotFoundError: If the model configuration file does not exist.

    This function checks for the existence of the model path and configuration
    file, updates the SPLIT_CONFIG dictionary with necessary paths, and then
    calls the run_predict function to perform the predictions.
    """
    PREDICTION_CONFIG["eopatches_folder"] = EOPATCHES_FOLDER
    results = run_predict(PREDICTION_CONFIG)
    LOGGER.info(f"Predictions for {len(results)} EOPatches completed.")


if __name__ == "__main__":
    PROJECT_DATA_ROOT = CONFIG['niva_project_data_root_inf']
    TILE_ID = CONFIG['TILE_ID']
    # Inferred constants
    PROJECT_DATA_ROOT = os.path.join(PROJECT_DATA_ROOT, TILE_ID)
    # the folders that will be created during the pipeline run
    EOPATCHES_FOLDER = os.path.join(PROJECT_DATA_ROOT, "eopatches")
    main_prediction(EOPATCHES_FOLDER)

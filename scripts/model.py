import logging
from filter import LogFileFilter
import sys
import os
import numpy as np
import tensorflow as tf
from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.metrics import MCCMetric, MeanIoU
from eoflow.models.losses import TanimotoDistanceLoss
from eoflow.models.segmentation_unets import ResUnetA
import argparse

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Constants
INPUT_SHAPE = [256, 256, 4]
MODEL_NAME = 'resunet-a'

# Model configuration (should not be changed)
MODEL_CONFIG = {
    "learning_rate": 0.0001,
    "n_layers": 3,
    "n_classes": 2,
    "keep_prob": 0.8,
    "features_root": 32,
    "conv_size": 3,
    "conv_stride": 1,
    "dilation_rate": [1, 3, 15, 31],
    "deconv_size": 2,
    "add_dropout": True,
    "add_batch_norm": False,
    "use_bias": False,
    "bias_init": 0.0,
    "padding": "SAME",
    "pool_size": 3,
    "pool_stride": 2,
    "prediction_visualization": True,
    "class_weights": None
}


@tf.function
def mcc_metric(y_t, y_p, threshold=0.5):
    # Matthew Correlation Coefficient implementation
    # https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
    y_true = y_t[..., -1]
    y_pred = y_p[..., -1]
    predicted = tf.cast(tf.greater(y_pred, threshold), tf.float32)
    true_pos = tf.math.count_nonzero(predicted * y_true)
    true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
    false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
    x = tf.cast((true_pos + false_pos) * (true_pos + false_neg)
                * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
    return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)


def initialise_model(input_shape, model_config, chkpt_folder=None):
    """
    Initializes and compiles a model for field delineation.

    Args:
        input_shape (tuple): The shape of the input images.
        model_config (dict): Configuration parameters for the model.
        chkpt_folder (str, optional): Path to the folder containing model checkpoint.
        Defaults to None.

    Returns:
        model: The compiled model for image field delineation.
    """
    model = ResUnetA(model_config)
    model.build(dict(features=[None] + list(input_shape)))
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=model_config['learning_rate'])
    model.net.compile(
        loss={'extent': TanimotoDistanceLoss(from_logits=False),
              'boundary': TanimotoDistanceLoss(from_logits=False),
              'distance': TanimotoDistanceLoss(from_logits=False)},
        optimizer=optimizer,
        metrics=[
            segmentation_metrics['accuracy'](),
            tf.keras.metrics.MeanIoU(
                num_classes=2,
                sparse_y_true=False,
                sparse_y_pred=False),
            mcc_metric
        ])
    if chkpt_folder is not None:
        model.net.load_weights(f'{chkpt_folder}/model.ckpt').expect_partial()
        LOGGER.info(f'Model loaded from existing checkpoint in {chkpt_folder}')
    return model


def get_average_from_models(
        model_list: list[str],
        model_folder: str,
        model_name: str):
    """
    Calculates the average weights from a list of models and sets the average weights to the given average model.
    Saves the average model to a specified folder and returns the average model.

    Args:
        avg_model (ResUnetA): The average model to set the average weights to.
        model_list (list[str]): A list of model names.
        model_folder (str): The folder path to save the average model.
        model_name (str): The name of the average model.

    Returns:
        avg_model (ResUnetA): The average model with the average weights set.
        avg_model_path (str): The path to the folder where the average model is saved.

    Raises:
        None
    """
    avg_model = initialise_model(INPUT_SHAPE, MODEL_CONFIG)
    weights = [model.net.get_weights() for model in model_list]
    avg_weights = [np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                   for weights_list_tuple in zip(*weights)]
    avg_model.net.set_weights(avg_weights)
    avg_model_path = f'{model_folder}/avg_{model_name}'
    os.makedirs(avg_model_path, exist_ok=True)
    checkpoints_path = os.path.join(
        avg_model_path, 'checkpoints', 'model.ckpt')
    avg_model.net.save_weights(checkpoints_path)
    LOGGER.info(f'Average model saved to {avg_model_path}')
    return avg_model, avg_model_path


def load_models(master_folder_path: str):
    """
    Load models from the specified master folder path.

    Args:
        master_folder_path (str): The path to the master folder.
        The master folder should contain sub-folders for each fold, each containing a checkpoints folder.
        Should have the following structure:
        .
        ├── fold_0_resunet-a
        │   ├── checkpoints
        │   │   ├── checkpoint
        │   │   ├── model.ckpt.data-00000-of-00001
        │   │   └── model.ckpt.index
        │   ├── ...
        ├── fold_1_resunet-a
        ├── fold_2_resunet-a
        ├── ...
        └── fold_N_resunet-a

    Returns:
        list: A list of loaded models.

    Raises:
        FileNotFoundError: If the master folder path does not exist.
    """
    if not os.path.exists(master_folder_path):
        raise FileNotFoundError(
            f'Master folder path {master_folder_path} does not exist')
    fold_folders = [f.path for f in os.scandir(
        master_folder_path) if f.is_dir() and f.name.startswith('fold')]
    LOGGER.info(
        f'Found {len(fold_folders)} fold folders in {master_folder_path}:\n{fold_folders}')
    model_list = []
    for fold_folder in fold_folders:
        checkpoint_path = os.path.join(fold_folder, 'checkpoints')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoints folder {checkpoint_path} does not exist for fold folder {fold_folder}')
        model = initialise_model(INPUT_SHAPE, MODEL_CONFIG, checkpoint_path)
        model_list.append(model)
    return model_list


def main():
    parser = argparse.ArgumentParser(
        description="Model Averaging for ResUnet-A Models")
    subparsers = parser.add_subparsers(dest='command')
    average_parser = subparsers.add_parser('average', help="Average models")
    average_parser.add_argument(
        'path', type=str, help="Path to the master folder containing model checkpoints")
    args = parser.parse_args()

    if args.command == 'average':
        master_folder_path = args.path
        model_list = load_models(master_folder_path)
        get_average_from_models(model_list, master_folder_path, MODEL_NAME)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

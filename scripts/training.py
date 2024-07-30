import os
import json
import logging
from datetime import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path
import pytz
import sys

from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import TanimotoDistanceLoss
from eoflow.models.segmentation_unets import ResUnetA
from functools import reduce
from filter import LogFileFilter

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Script parameters
TF_PROFILING = True
UPDATE_FREQ = 'epoch'
TIMEZONE = pytz.timezone('Europe/Paris')

# Paths parameters
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
DATASET_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/datasets/')
MODEL_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/model/')
CHKPT_FOLDER = None
ENABLE_DATA_SHARDING = True

# Model hyperparameters
ITERATIONS_PER_EPOCH = 30
NUM_EPOCHS = 2
INPUT_SHAPE = [256, 256, 4]
N_CLASSES = 2
BATCH_SIZE = 8
MODEL_NAME = "resunet-a"
N_FOLDS = 10

# Model configuration
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


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    CustomTensorBoard is a subclass of tf.keras.callbacks.TensorBoard.
    It extends the functionality of TensorBoard by adding profiling capabilities.

    Args:
        log_dir (str): The directory where the TensorBoard logs will be saved.
        **kwargs: Additional keyword arguments to be passed to the base class constructor.

    Attributes:
        log_dir (str): The directory where the TensorBoard logs will be saved.

    Methods:
        on_train_begin(logs=None): Called at the beginning of training.
        on_train_end(logs=None): Called at the end of training.
    """

    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        If TF_PROFILING is enabled, starts the profiler and logs a message.

        Args:
            logs (dict): Dictionary of logs, containing the current training metrics.
        """
        super().on_train_begin(logs)
        if TF_PROFILING:
            tf.profiler.experimental.start(self.log_dir)
            LOGGER.info(f"Profiler started at {self.log_dir}")

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        If TF_PROFILING is enabled, stops the profiler and logs a message.

        Args:
            logs (dict): Dictionary of logs, containing the final training metrics.
        """
        super().on_train_end(logs)
        if TF_PROFILING:
            tf.profiler.experimental.stop()
            LOGGER.info(f"Profiler stopped and data saved to {self.log_dir}")


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

    model.net.compile(
        loss={'extent': TanimotoDistanceLoss(from_logits=False),
              'boundary': TanimotoDistanceLoss(from_logits=False),
              'distance': TanimotoDistanceLoss(from_logits=False)},
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=model_config['learning_rate']),
        metrics=[segmentation_metrics['accuracy'](),
                 tf.keras.metrics.MeanIoU(num_classes=2)])

    if chkpt_folder is not None:
        model.net.load_weights(f'{chkpt_folder}/model.ckpt')

    return model


def initialise_callbacks(model_folder, model_name, fold, model_config):
    """
    Initialize callbacks for model training.

    Args:
        model_folder (str): The folder path where the model will be saved.
        model_name (str): The name of the model.
        fold (int): The fold number.
        model_config (dict): The model configuration.

    Returns:
        tuple: A tuple containing the model path and a list of callbacks.

    """
    timestamp = datetime.now(TIMEZONE)
    now = f"{timestamp.day}-{timestamp.month}-{timestamp.hour}-{timestamp.minute}"
    creator = os.getlogin()
    model_path = f'{model_folder}/{creator}-fold-{fold}_{now}'

    os.makedirs(model_path, exist_ok=True)
    logs_path = os.path.join(model_path, 'logs')
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')

    tensorboard_callback = CustomTensorBoard(
        log_dir=logs_path, update_freq=UPDATE_FREQ)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoints_path, save_best_only=True, save_freq='epoch', save_weights_only=True)

    with open(f'{model_path}/model_cfg.json', 'w') as jfile:
        json.dump(model_config, jfile)

    callbacks = [tensorboard_callback, checkpoint_callback]
    return model_path, callbacks


def set_auto_shard_policy(dataset):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return dataset.with_options(options)


def load_and_process_dataset(dataset_folder, fold, batch_size):
    dataset_path = os.path.join(dataset_folder, f'fold_{fold}')
    dataset = tf.data.Dataset.load(dataset_path)
    dataset = dataset.batch(batch_size)
    if ENABLE_DATA_SHARDING:
        dataset = set_auto_shard_policy(dataset)
    return dataset


def train_k_folds(dataset_folder, model_folder, chkpt_folder, input_shape,
                  batch_size, iterations_per_epoch, num_epochs,
                  model_name, n_folds, model_config):

    LOGGER.info('Loading K TF datasets')
    ds_folds = []
    for fold in range(1, n_folds + 1):
        dataset = load_and_process_dataset(dataset_folder, fold, batch_size)
        ds_folds.append(dataset)

    folds = list(range(n_folds))
    folds_ids_list = [(folds[:nf] + folds[1 + nf:], [nf]) for nf in folds]
    np.random.seed()

    models = []
    model_paths = []

    strategy = tf.distribute.MirroredStrategy()
    num_workers = strategy.num_replicas_in_sync
    devices = strategy.extended.worker_devices
    LOGGER.info(
        f"Number of devices (workers): {num_workers}\nDevices: {devices}")

    for training_ids, testing_id in folds_ids_list:
        left_out_fold = testing_id[0] + 1
        LOGGER.info(f'Training model for left-out fold {left_out_fold}')

        fold_val = np.random.choice(training_ids)
        folds_train = [tid for tid in training_ids if tid != fold_val]
        LOGGER.info(
            f'\tTrain folds {folds_train}, Val fold: {fold_val}, Test fold: {testing_id[0]}')

        ds_folds_train = [ds_folds[tid] for tid in folds_train]
        ds_train = reduce(tf.data.Dataset.concatenate, ds_folds_train)
        ds_val = ds_folds[fold_val]
        ds_train = ds_train.repeat()

        with strategy.scope():
            model = initialise_model(
                input_shape, model_config, chkpt_folder=chkpt_folder)
            model_path, callbacks = initialise_callbacks(
                model_folder, model_name, left_out_fold, model_config)
            LOGGER.info(f'\tTraining model, writing to {model_path}')
            try:
                model.net.fit(ds_train,
                              validation_data=ds_val,
                              epochs=num_epochs,
                              steps_per_epoch=iterations_per_epoch,
                              callbacks=callbacks)
            except Exception as e:
                LOGGER.error(f"Exception during training: {e}")
            models.append(model)
            model_paths.append(model_path)
            LOGGER.info(
                f'\tModel trained and saved to {model_path} for left out fold {left_out_fold}')

    LOGGER.info('Create average model')
    weights = [model.net.get_weights() for model in models]
    avg_weights = [np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                   for weights_list_tuple in zip(*weights)]

    avg_model = initialise_model(input_shape, model_config)
    avg_model.net.set_weights(avg_weights)

    now = datetime.now(TIMEZONE).isoformat(
        sep='-', timespec='seconds').replace(':', '-')
    model_path = f'{model_folder}/{model_name}_avg_{now}'

    LOGGER.info('Save average model to local path')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')
    with open(f'{model_path}/model_cfg.json', 'w+') as jfile:
        json.dump(model_config, jfile)
    avg_model.net.save_weights(checkpoints_path)

    for _, testing_id in folds_ids_list:
        left_out_fold = testing_id[0] + 1
        LOGGER.info(f'Evaluating model on left-out fold {left_out_fold}')
        model = models[testing_id[0]]
        model.net.evaluate(ds_folds[testing_id[0]])
        LOGGER.info(
            f'Evaluating average model on left-out fold {left_out_fold}')
        avg_model.net.evaluate(ds_folds[testing_id[0]])
        LOGGER.info('\n\n')


if __name__ == '__main__':
    train_k_folds(DATASET_FOLDER, MODEL_FOLDER, CHKPT_FOLDER, INPUT_SHAPE,
                  BATCH_SIZE, ITERATIONS_PER_EPOCH, NUM_EPOCHS, MODEL_NAME,
                  N_FOLDS, MODEL_CONFIG)

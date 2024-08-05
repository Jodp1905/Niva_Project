import os
import json
import logging
from datetime import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path
import pytz
import sys
import psutil
import time

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

# Model hyperparameters
HYPER_PARAM_CONFIG = {
    "iterations_per_epoch": 50,
    "num_epochs": 5,
    "batch_size": 4,
    "n_classes": 2,
    "n_folds": 10,
    "tf_full_profiling": False,
    "prefetch_data": False,
    "enable_data_sharding": True,
    "chkpt_folder": None,
    "input_shape": [256, 256, 4],
    "model_name": "resunet-a"
}

# Timezone parameters
UPDATE_FREQ = 'epoch'
TIMEZONE = pytz.timezone('Europe/Paris')

# Paths parameters
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
DATASET_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/datasets/')
MODEL_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/model/')

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
        if HYPER_PARAM_CONFIG['tf_full_profiling']:
            tf.profiler.experimental.start(self.log_dir)
            LOGGER.info(f"Full Profiler started at {self.log_dir}")

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        If TF_PROFILING is enabled, stops the profiler and logs a message.

        Args:
            logs (dict): Dictionary of logs, containing the final training metrics.
        """
        if HYPER_PARAM_CONFIG['tf_full_profiling']:
            tf.profiler.experimental.stop()
            LOGGER.info(
                f"Full Profiler stopped and data saved to {self.log_dir}")
        super().on_train_end(logs)


class PerformanceLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.epoch_start_time = None
        self.batch_start_time = None

    def on_train_begin(self, logs=None):
        mem_info = psutil.virtual_memory()
        total_memory_gb = mem_info.total / (1024 ** 3)  # in GB

        with self.writer.as_default():
            tf.summary.scalar('Memory/Total_memory_GB',
                              total_memory_gb, step=0)
            self.writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_memory_gb = mem_info.rss / (1024 ** 3)  # in GB
        vms_memory_gb = mem_info.vms / (1024 ** 3)  # in GB

        model_size_gb = self.calculate_model_size_gb()

        with self.writer.as_default():
            tf.summary.scalar(
                'Performance/Epoch_duration_seconds', epoch_duration, step=epoch)
            tf.summary.scalar('Memory/RSS_memory_GB',
                              rss_memory_gb, step=epoch)
            tf.summary.scalar('Memory/VMS_memory_GB',
                              vms_memory_gb, step=epoch)
            tf.summary.scalar('Model/Model_size_GB', model_size_gb, step=epoch)
            self.writer.flush()

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        batch_duration = time.time() - self.batch_start_time
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_memory_gb = mem_info.rss / (1024 ** 3)  # in GB
        vms_memory_gb = mem_info.vms / (1024 ** 3)  # in GB

        model_size_gb = self.calculate_model_size_gb()

        with self.writer.as_default():
            tf.summary.scalar(
                'Performance/Batch_duration_seconds', batch_duration, step=batch)
            tf.summary.scalar('Memory/RSS_memory_GB',
                              rss_memory_gb, step=batch)
            tf.summary.scalar('Memory/VMS_memory_GB',
                              vms_memory_gb, step=batch)
            tf.summary.scalar('Model/Model_size_GB', model_size_gb, step=batch)
            self.writer.flush()

    def calculate_model_size_gb(self):
        model_size_bytes = sum([tf.size(variable).numpy(
        ) * variable.dtype.size for variable in self.model.trainable_weights])
        model_size_gb = model_size_bytes / (1024 ** 3)  # in GB
        return model_size_gb


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


def initialise_callbacks(model_folder, fold):
    """
    Initialize callbacks for model training.

    Args:
        model_folder (str): The folder path where the model will be saved.
        fold (int): The fold number.

    Returns:
        tuple: A tuple containing the model path and a list of callbacks.

    """
    timestamp = datetime.now(TIMEZONE)
    now = f"{timestamp.day}-{timestamp.month}-{timestamp.hour}-{timestamp.minute}"
    model_path = f'{model_folder}/fold-{fold}_{now}'

    os.makedirs(model_path, exist_ok=True)
    logs_path = os.path.join(model_path, 'logs')
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')

    tensorboard_callback = CustomTensorBoard(
        log_dir=logs_path,
        update_freq=UPDATE_FREQ,
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoints_path,
        save_best_only=True,
        save_freq='epoch',
        save_weights_only=True)

    performance_callback = PerformanceLoggingCallback(log_dir=logs_path)

    callbacks = [tensorboard_callback,
                 checkpoint_callback, performance_callback]
    return model_path, callbacks, logs_path


def set_auto_shard_policy(dataset):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return dataset.with_options(options)


def load_and_process_dataset(dataset_folder, fold, batch_size):
    dataset_path = os.path.join(dataset_folder, f'fold_{fold}')
    dataset = tf.data.Dataset.load(dataset_path)
    dataset = dataset.batch(batch_size)
    if HYPER_PARAM_CONFIG['enable_data_sharding']:
        dataset = set_auto_shard_policy(dataset)
    if HYPER_PARAM_CONFIG['prefetch_data']:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def train_k_folds(dataset_folder, model_folder, chkpt_folder, input_shape,
                  batch_size, iterations_per_epoch, num_epochs,
                  model_name, n_folds, model_config):

    # Dump hyperparameters to json
    with open(f'{model_folder}/hyperparameters.json', 'w') as jfile:
        json.dump(HYPER_PARAM_CONFIG, jfile)

    # Dump model configuration to json
    with open(f'{model_folder}/model_cfg.json', 'w') as jfile:
        json.dump(model_config, jfile)

    training_full_start_time = time.time()
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

        fold_val = np.random.choice(training_ids)
        folds_train = [tid for tid in training_ids if tid != fold_val]
        LOGGER.info(
            f'\tTrain folds {folds_train}, Val fold: {fold_val}, Test fold: {testing_id[0]}')

        ds_folds_train = [ds_folds[tid] for tid in folds_train]
        ds_train = reduce(tf.data.Dataset.concatenate, ds_folds_train)
        ds_val = ds_folds[fold_val]
        # TODO repeat without argument repeats indefinitely, memory leak
        ds_train = ds_train.repeat()

        with strategy.scope():
            training_start_time = time.time()
            model = initialise_model(
                input_shape, model_config, chkpt_folder=chkpt_folder)
            model_path, callbacks, logs_path = initialise_callbacks(
                model_folder, testing_id[0])
            LOGGER.info(f'\tTraining model, writing to {model_path}')
            try:
                model.net.fit(ds_train,
                              validation_data=ds_val,
                              epochs=num_epochs,
                              # TODO Perhaps don't set steps_per_epoch and let it iterate over the dataset
                              steps_per_epoch=iterations_per_epoch,
                              callbacks=callbacks)
            except Exception as e:
                LOGGER.error(f"Exception during training: {e}")
            training_end_time = time.time()
            models.append(model)
            model_paths.append(model_path)
            LOGGER.info(
                f'\tModel trained and saved to {model_path} for evaluation on fold {testing_id[0]}'
                f'\n\tTraining time: {training_end_time - training_start_time} seconds')

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
    avg_model.net.save_weights(checkpoints_path)

    for _, testing_id in folds_ids_list:
        LOGGER.info(f'Evaluating model on left-out fold {testing_id[0]}')
        model = models[testing_id[0]]
        model.net.evaluate(ds_folds[testing_id[0]])
        LOGGER.info(
            f'Evaluating average model on left-out fold {testing_id[0]}')
        avg_model.net.evaluate(ds_folds[testing_id[0]])
        LOGGER.info('\n\n')
    train_full_end_time = time.time()
    LOGGER.info(
        f'Training all models and average model took {train_full_end_time - training_full_start_time} seconds')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        LOGGER.error('Usage: python training.py <model_name>')
        exit(1)
    model_name = sys.argv[1]
    model_folder = os.path.join(MODEL_FOLDER, model_name)
    os.makedirs(model_folder, exist_ok=True)
    train_k_folds(DATASET_FOLDER,
                  model_folder,
                  HYPER_PARAM_CONFIG['chkpt_folder'],
                  HYPER_PARAM_CONFIG['input_shape'],
                  HYPER_PARAM_CONFIG['batch_size'],
                  HYPER_PARAM_CONFIG['iterations_per_epoch'],
                  HYPER_PARAM_CONFIG['num_epochs'],
                  HYPER_PARAM_CONFIG['model_name'],
                  HYPER_PARAM_CONFIG['n_folds'],
                  MODEL_CONFIG)

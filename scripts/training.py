# autopep8: off
import os
import json
import logging
from pathlib import Path
import pytz
import sys
import psutil
import time
import pandas as pd
from enum import Enum
from functools import reduce
from filter import LogFileFilter
import json
import socket
import tensorflow as tf

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


# Configure TensorFlow strategies at the earliest


class TrainingType(Enum):
    # Uses MirroredStrategy for single worker
    SingleWorker = 1
    # Uses MultiWorkerMirroredStrategy for multiple workers
    MultiWorker = 2


TRAINING_TYPE_ENV = os.getenv('TRAINING_TYPE', TrainingType.SingleWorker.name)
STRATEGY = None
TF_CONFIG, TF_CONFIG_DICT = None, None
if TRAINING_TYPE_ENV == TrainingType.SingleWorker.name:
    LOGGER.info("SingleWorker selected, using MirroredStrategy")
    STRATEGY = tf.distribute.MirroredStrategy()
    num_workers = STRATEGY.num_replicas_in_sync
    devices = STRATEGY.extended.worker_devices
    hostname = socket.gethostname()
    LOGGER.info(
        f"\n\n========================================"
        f" Strategy Summary "
        f"========================================\n"
        f"MirroredStrategy selected with {num_workers} workers on {hostname}\n"
        f"Devices: {devices}")
elif TRAINING_TYPE_ENV == TrainingType.MultiWorker.name:
    LOGGER.info("MultiWorker selected, using MultiWorkerMirroredStrategy")
    TF_CONFIG = os.getenv('TF_CONFIG')
    if TF_CONFIG is None:
        LOGGER.error('TF_CONFIG environment variable not set')
        exit(1)
    TF_CONFIG_DICT = json.loads(TF_CONFIG)
    STRATEGY = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CollectiveCommunication.RING),
        cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver()
    )
    num_workers = STRATEGY.num_replicas_in_sync
    devices = STRATEGY.extended.worker_devices
    hostname = socket.gethostname()
    LOGGER.info(
        f"\n\n========================== Strategy Summary ==========================\n"
        f"[{hostname}] MultiWorkerMirroredStrategy selected with {num_workers} workers.\n"
        f"Devices: {devices}\n"
        f"TF_CONFIG: {TF_CONFIG_DICT}")
else:
    LOGGER.error(
        f"Invalid training type: {TRAINING_TYPE_ENV}.\n"
        f"Must be one of: {', '.join(TrainingType.__members__.keys())}")
    exit(1)

# Model related imports, have to be done after setting up the strategy
from model import get_average_from_models, initialise_model
from model import INPUT_SHAPE, MODEL_CONFIG
from create_datasets import get_dataset

# autopep8: on

# Training hyperparameters
TRAINING_CONFIG = {
    # Number of full passes through the dataset
    "num_epochs": int(os.getenv('NUM_EPOCHS', 20)),
    # Number of samples processed per batch
    "batch_size": int(os.getenv('BATCH_SIZE', 8)),
    # Number of classes in the dataset (2 for binary labels)
    "n_classes": 2,
    # Number of batches processed per epoch, default to -1 (full dataset)
    "iterations_per_epoch": int(os.getenv('ITERATIONS_PER_EPOCH', -1)),
    # Use dataset or npz files
    "use_npz": False,
    # Enable detailed profiling/tracing with TensorBoard
    # TODO : full_profiling eats all available memory, implement periodic flushing
    "tf_full_profiling": False,
    # Enable data prefetching runtime optimization
    "prefetch_data": True,
    # Enable data sharding runtime optimization
    "enable_data_sharding": False,
    # Path to the folder containing model checkpoint
    "chkpt_folder": None,
}

# Timezone parameters
UPDATE_FREQ = 'epoch'
TIMEZONE = pytz.timezone('Europe/Paris')

# Paths parameters
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
DATASET_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/datasets/')
MODEL_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/models/')
NPZ_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/npz_files/')


class PerformanceLoggingCallback(tf.keras.callbacks.Callback):
    """
    Callback for logging performance metrics during training.

    Args:
        model_path (str): The path to save the model and log files.

    Attributes:
        model_path (str): The path to save the model and log files.
        writer (tf.summary.FileWriter): The file writer for writing summary logs.
        epoch_start_time (float): The start time of the current epoch.
        batch_start_time (float): The start time of the current batch.
        batch_data (list): List to store batch data.
        epoch_data (list): List to store epoch data.
        model_size (float): The size of the model in GB.

    Methods:
        on_train_begin(logs=None):
            Called at the beginning of training.
            Logs the total memory and model size.

        on_epoch_begin(epoch, logs=None):
            Called at the beginning of each epoch.
            Sets the start time of the epoch.

        on_epoch_end(epoch, logs=None):
            Called at the end of each epoch.
            Logs the epoch duration, RSS memory, and VMS memory.
            Appends epoch data to epoch_data list.

        on_train_batch_begin(batch, logs=None):
            Called at the beginning of each training batch.
            Sets the start time of the batch.

        on_train_batch_end(batch, logs=None):
            Called at the end of each training batch.
            Logs the batch duration.
            Appends batch data to batch_data list.

        on_train_end(logs=None):
            Called at the end of training.
            Saves the epoch data and batch data to CSV files.

        calculate_model_size_gb():
            Calculates the size of the model in GB.
    """

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.writer = tf.summary.create_file_writer(model_path)
        self.epoch_start_time = None
        self.batch_start_time = None
        self.batch_data = []
        self.epoch_data = []
        self.model_size = None

    def on_train_begin(self, logs=None):
        mem_info = psutil.virtual_memory()
        total_memory_gb = mem_info.total / (1024 ** 3)  # in GB
        model_size_gb = self.calculate_model_size_gb()
        self.model_size = model_size_gb

        with self.writer.as_default():
            tf.summary.scalar('Memory/Total_memory_GB',
                              total_memory_gb, step=0)
            tf.summary.scalar('Model_size_GB',
                              model_size_gb, step=0)
            self.writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_memory_gb = mem_info.rss / (1024 ** 3)  # in GB
        vms_memory_gb = mem_info.vms / (1024 ** 3)  # in GB

        with self.writer.as_default():
            tf.summary.scalar(
                'Performance/Epoch_duration_seconds', epoch_duration, step=epoch)
            tf.summary.scalar('Memory/RSS_memory_GB',
                              rss_memory_gb, step=epoch)
            tf.summary.scalar('Memory/VMS_memory_GB',
                              vms_memory_gb, step=epoch)
            self.writer.flush()

        epoch_data_entry = {
            'epoch': epoch,
            'epoch_duration': epoch_duration,
            'rss_memory_gb': rss_memory_gb,
            'vms_memory_gb': vms_memory_gb,
            **(logs or {})
        }
        self.epoch_data.append(epoch_data_entry)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        batch_duration = time.time() - self.batch_start_time

        batch_data_entry = {
            'batch': batch,
            'batch_duration': batch_duration,
            **(logs or {})
        }
        self.batch_data.append(batch_data_entry)

    def on_train_end(self, logs=None):
        epoch_data_path = os.path.join(self.model_path, 'epoch_data.csv')
        pd.DataFrame(self.epoch_data).to_csv(epoch_data_path, index=False)
        batch_data_path = os.path.join(self.model_path, 'batch_data.csv')
        pd.DataFrame(self.batch_data).to_csv(batch_data_path, index=False)
        super().on_train_end(logs)

    def calculate_model_size_gb(self):
        model_size_bytes = sum([tf.size(variable).numpy(
        ) * variable.dtype.size for variable in self.model.trainable_weights])
        model_size_gb = model_size_bytes / (1024 ** 3)  # in GB
        return model_size_gb


def initialise_callbacks(model_path):
    """
    Initialize callbacks for model training.

    Args:
        model_path (str): The path to save the model and log files.

    Returns:
        list: List of callbacks for model training.
    """
    logs_path = os.path.join(model_path, 'logs')
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_path,
        update_freq=UPDATE_FREQ,
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoints_path,
        save_best_only=True,
        save_freq='epoch',
        save_weights_only=True)

    performance_callback = PerformanceLoggingCallback(model_path=model_path)

    callbacks = [tensorboard_callback, checkpoint_callback,
                 performance_callback]
    return callbacks


def set_auto_shard_policy(dataset):
    """
    Sets the auto shard policy for the given dataset to DATA.

    Parameters:
        dataset (tf.data.Dataset): The input dataset.

    Returns:
        tf.data.Dataset: The dataset with the auto shard policy set.
    """
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return dataset.with_options(options)


def load_and_process_dataset(dataset_path, batch_size):
    """
    Loads and processes a dataset from the specified folder and fold.
    Apply runtime optimizations : batch, shard, prefetch.

    Args:
        dataset_path (str): The path to the dataset folder.
        batch_size (int): The batch size for the dataset.

    Returns:
        tf.data.Dataset: The loaded and processed dataset.
    """
    dataset = tf.data.Dataset.load(dataset_path)
    # batch
    dataset = dataset.batch(batch_size)
    # shard
    if TRAINING_CONFIG['enable_data_sharding']:
        dataset = set_auto_shard_policy(dataset)
    # prefetch
    if TRAINING_CONFIG['prefetch_data']:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_and_process_npz(dataset_path, batch_size, fold_type):
    """
    Loads and processes a dataset from NPZ files.

    Args:
        dataset_path (str): The path to the dataset file in NPZ format.
        batch_size (int): The size of the batches to create from the dataset.
        fold_type (str): The type of fold to use for the dataset (e.g., 'train', 'validation', 'test').

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object that has been batched, optionally sharded, and prefetched.
    """
    dataset = get_dataset(dataset_path, fold_type)
    # batch
    dataset = dataset.batch(batch_size)
    # shard
    if TRAINING_CONFIG['enable_data_sharding']:
        dataset = set_auto_shard_policy(dataset)
    # prefetch
    if TRAINING_CONFIG['prefetch_data']:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def get_dataset_size(dataset: tf.data.Dataset,
                     img_count: bool = True) -> int:
    """
    Returns the number of elements in a dataset.

    Args:
        dataset (tf.data.Dataset): The dataset.
        img_count (bool, optional): If True, returns the number of images.
            If False, returns the number of samples. Defaults to True.

    Returns:
        int: The number of elements in the dataset.
    """
    def add_one(x, _):
        return x + 1
    num_samples = dataset.reduce(0, add_one).numpy()
    if img_count is False:
        return num_samples
    else:
        batch_size_global = TRAINING_CONFIG['batch_size']
        num_imgs = num_samples * batch_size_global
        return num_imgs


def check_positive_integers(**kwargs):
    failed_args = {name: value for name, value in kwargs.items() if not (
        isinstance(value, int) and value >= 0)}
    return failed_args


def training_main(
        strategy: tf.distribute.Strategy,
        model_folder: str,
        model_config: dict,
        input_shape: tuple,
        chkpt_folder: str,
        num_epochs: int,
        iterations_per_epoch: int,
        batch_size: int):
    """
    Trains a model using k-fold cross-validation, and performs evaluation on the left-out fold.
    At the end, an average model is created and evaluated on all folds.

    Args:
        strategy (tf.distribute.Strategy): The strategy for distributed training.
        model_folder (str): The path to the folder where the trained models will be saved.
        model_config (dict): The configuration of the model.
        input_shape (tuple): The shape of the input data.
        chkpt_folder (str): The path to the folder where the model checkpoints will be saved.
        num_epochs (int): The number of epochs to train for.
        iterations_per_epoch (int): The number of iterations per epoch.
        batch_size (int): The batch size for training.

    Returns:
        None
    """
    training_full_start_time = time.time()

    # Arguments sanity check
    failed_args = check_positive_integers(
        batch_size=batch_size,
        num_epochs=num_epochs)
    if iterations_per_epoch is not None and iterations_per_epoch <= 0:
        failed_args['iterations_per_epoch'] = iterations_per_epoch
    if failed_args:
        failed_msg = ', '.join(
            f"{name}={value}({type(value)})" for name, value in failed_args.items())
        raise ValueError(
            f"The following arguments must be positive integers: {failed_msg}")

    # Dump hyperparameters and model configuration to json
    with open(f'{model_folder}/hyperparameters.json', 'w') as jfile:
        json.dump(TRAINING_CONFIG, jfile, indent=4)
    with open(f'{model_folder}/model_cfg.json', 'w') as jfile:
        json.dump(model_config, jfile, indent=4)

    # Load datasets
    LOGGER.info(
        f'\n\n========================================'
        f' Dataset Loading '
        f'=========================================')
    datasets = {}
    sample_count = 0
    for fold in ["train", "val", "test"]:
        if TRAINING_CONFIG['use_npz']:
            dataset_path = os.path.join(NPZ_FOLDER, fold)
            dataset = load_and_process_npz(dataset_path, batch_size, fold)
        else:
            dataset_path = os.path.join(DATASET_FOLDER, fold)
            dataset = load_and_process_dataset(dataset_path, batch_size)
        dataset_size = get_dataset_size(dataset, img_count=False)
        LOGGER.info(f'Loaded fold {fold} with {dataset_size} samples,'
                    f' or {dataset_size * batch_size} images')
        sample_count += dataset_size
        datasets[fold] = dataset
    img_count = sample_count * batch_size
    LOGGER.info(f'Loaded {sample_count} samples in total. With a batch '
                f'size of {batch_size}, this corresponds to {img_count} images')

    # Dataset splitting
    ds_train = datasets['train']
    ds_val = datasets['val']
    ds_test = datasets['test']

    if iterations_per_epoch is not None:
        # repeat dataset if iterations_per_epoch is set
        ds_train = ds_train.repeat()

    # Perform fitting using the strategy scope
    with strategy.scope():
        init_start = time.time()
        model = initialise_model(
            input_shape, model_config, chkpt_folder=chkpt_folder)
        LOGGER.info(
            f'\n========================================='
            f' Model Summary '
            f'==========================================')
        model.net.summary()
        model.net.count_params()
        callbacks = initialise_callbacks(model_folder)
        init_end = time.time()
        init_duration = init_end - init_start
        # Fit model
        fitting_start_time = time.time()
        try:
            if iterations_per_epoch is not None:
                # fit with steps_per_epoch as we repeat the dataset
                model.net.fit(ds_train,
                              validation_data=ds_val,
                              epochs=num_epochs,
                              steps_per_epoch=iterations_per_epoch,
                              callbacks=callbacks)
            else:
                # fit without steps_per_epoch and iterate over the full dataset
                model.net.fit(ds_train,
                              validation_data=ds_val,
                              epochs=num_epochs,
                              callbacks=None)
        except Exception as e:
            raise Exception(f'Error during model fitting') from e
        fitting_end_time = time.time()
        fitting_duration = fitting_end_time - fitting_start_time

        # Evaluate model on testing dataset
        LOGGER.info(f'Evaluating model on testing dataset')
        testing_start_time = time.time()
        try:
            evaluation = model.net.evaluate(ds_test)
        except Exception as e:
            raise Exception(f'Error during evaluation') from e
        testing_end_time = time.time()
        testing_duration = testing_end_time - testing_start_time
        evaluation_dict = dict(zip(model.net.metrics_names, evaluation))
        evaluation_path = os.path.join(model_folder, 'evaluation.json')
        with open(evaluation_path, 'w') as jfile:
            json.dump(evaluation_dict, jfile, indent=4)
        LOGGER.info(
            f'\tEvaluation results saved to {model_folder}/evaluation.json')

        # Registering fold configuration and training duration
        fold_duration = init_duration + fitting_duration + testing_duration
        model_size = callbacks[-1].model_size
        LOGGER.info(f'\n'
                    f'Model init duration: {init_duration} seconds \n'
                    f'Model fitting duration: {fitting_duration} seconds \n'
                    f'Model testing duration: {testing_duration} seconds \n')
        fold_infos = {
            'model_size_gb': model_size,
            'fold_duration': fold_duration,
            'init_duration': init_duration,
            'fitting_duration': fitting_duration,
            'testing_duration': testing_duration
        }
        fold_data_path = os.path.join(model_folder, 'fold_infos.json')
        with open(fold_data_path, 'w') as jfile:
            json.dump(fold_infos, jfile, indent=4)

    # Save training duration
    train_full_end_time = time.time()
    training_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(train_full_end_time - training_full_start_time))
    LOGGER.info(f'Full training duration : {training_time_str}')


if __name__ == '__main__':
    # Check environment variables
    if NIVA_PROJECT_DATA_ROOT is None:
        LOGGER.error('NIVA_PROJECT_DATA_ROOT environment variable not set')
        exit(1)
    # Check arguments
    if len(sys.argv) != 2:
        LOGGER.error('Usage: python training.py <run_name>')
        exit(1)
    run_name = sys.argv[1]
    # Set up model folder
    model_folder = os.path.join(MODEL_FOLDER, run_name)
    os.makedirs(model_folder, exist_ok=True)
    if TRAINING_CONFIG['iterations_per_epoch'] == -1:
        TRAINING_CONFIG['iterations_per_epoch'] = None
    training_main(
        strategy=STRATEGY,
        model_folder=model_folder,
        model_config=MODEL_CONFIG,
        input_shape=INPUT_SHAPE,
        chkpt_folder=TRAINING_CONFIG['chkpt_folder'],
        num_epochs=TRAINING_CONFIG['num_epochs'],
        iterations_per_epoch=TRAINING_CONFIG['iterations_per_epoch'],
        batch_size=TRAINING_CONFIG['batch_size'])
    LOGGER.info('Training completed')

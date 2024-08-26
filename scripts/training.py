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
import pandas as pd

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
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Model hyperparameters
HYPER_PARAM_CONFIG = {
    # Number of full passes through the dataset
    "num_epochs": 20,
    # Number of samples processed per batch
    "batch_size": 8,
    # Number of classes in the dataset (2 for binary labels)
    "n_classes": 2,
    # Number of folds for cross-validation
    "n_folds": 10,
    # If True, repeats dataset and use iterations_per_epoch during model fit
    "repeat_dataset": False,
    # Number of batches processed per epoch, used only if repeat_dataset is True
    # Should correspond to the number of samples in a training fold dataset
    "iterations_per_epoch": ((9 / 10) * (2000 * 6)) // 4,
    # TODO : full_profiling eats all available memory, implement periodic flushing
    # Enable detailed profiling/tracing with TensorBoard
    "tf_full_profiling": False,
    # Enable data prefetching runtime optimization
    "prefetch_data": True,
    # Enable data sharding runtime optimization
    "enable_data_sharding": True,
    # Path to the folder containing model checkpoint
    "chkpt_folder": None,
    # Shape of the input images
    "input_shape": [256, 256, 4],
    # Name of the model
    "model_name": "resunet-a"
}

# Timezone parameters
UPDATE_FREQ = 'epoch'
TIMEZONE = pytz.timezone('Europe/Paris')

# Paths parameters
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
DATASET_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/datasets/')
MODEL_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/model/')

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


def load_and_process_dataset(dataset_folder, fold, batch_size):
    """
    Loads and processes a dataset from the specified folder and fold.
    Apply runtime optimizations : batch, shard, prefetch.

    Args:
        dataset_folder (str): The path to the dataset folder.
        fold (int): The fold number.
        batch_size (int): The batch size for the dataset.

    Returns:
        tf.data.Dataset: The loaded and processed dataset.
    """
    dataset_path = os.path.join(dataset_folder, f'fold_{fold}')
    dataset = tf.data.Dataset.load(dataset_path)
    # batch
    dataset = dataset.batch(batch_size)
    # shard
    if HYPER_PARAM_CONFIG['enable_data_sharding']:
        dataset = set_auto_shard_policy(dataset)
    # prefetch
    if HYPER_PARAM_CONFIG['prefetch_data']:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def prepare_fold_configurations(n_folds, model_path, seed=42):
    """
    Prepare fold configurations for n-fold cross-validation.

    Args:
        n_folds (int): Number of folds for cross-validation.
        model_path (str): Path to the model directory.

    Returns:
        list: List of dictionaries containing fold configurations.
            Each dictionary contains the following keys:
            - 'testing_fold': The index of the testing fold.
            - 'validation_fold': The index of the validation fold.
            - 'training_folds': List of indices of the training folds.
            - 'model_fold_path': Path to the model directory for the testing fold.
    """
    folds = list(range(n_folds))
    folds_ids_list = [folds[:nf] + folds[1 + nf:] for nf in folds]
    fold_configurations = []
    for i in range(n_folds):
        testing_fold = folds[i]
        model_fold_path = os.path.join(model_path, f'fold-{testing_fold}')
        os.makedirs(model_fold_path, exist_ok=True)
        np.random.seed(seed)
        validation_fold = int(np.random.choice(folds_ids_list[i]))
        training_folds = [tid for tid in folds_ids_list[i]
                          if tid != validation_fold]
        fold_entry = {
            'testing_fold': testing_fold,
            'validation_fold': validation_fold,
            'training_folds': training_folds,
            'model_fold_path': model_fold_path
        }
        fold_configurations.append(fold_entry)
    return fold_configurations


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
    num_samples = dataset.reduce(0, lambda x, _: x + 1).numpy()
    if img_count is False:
        return num_samples
    else:
        batch_size_global = HYPER_PARAM_CONFIG['batch_size']
        num_imgs = num_samples * batch_size_global
        return num_imgs


def train_k_folds(dataset_folder, model_folder, chkpt_folder, input_shape,
                  batch_size, iterations_per_epoch, num_epochs,
                  model_name, n_folds, model_config, start_fold=0):
    """
    Trains a model using k-fold cross-validation, and performs evaluation on the left-out fold.
    At the end, an average model is created and evaluated on all folds.

    Args:
        dataset_folder (str): The path to the folder containing the dataset.
        model_folder (str): The path to the folder where the trained models will be saved.
        chkpt_folder (str): The path to the folder where the model checkpoints will be saved.
        input_shape (tuple): The shape of the input data.
        batch_size (int): The batch size for training.
        iterations_per_epoch (int): The number of iterations per epoch.
        num_epochs (int): The number of epochs to train for.
        model_name (str): The name of the model.
        n_folds (int): The number of folds for cross-validation.
        model_config (dict): The configuration of the model.
        start_fold (int): The fold from which to resume the training (default is 0).

    Returns:
        None
    """
    training_full_start_time = time.time()

    # Dump hyperparameters and model configuration to json
    with open(f'{model_folder}/hyperparameters.json', 'w') as jfile:
        json.dump(HYPER_PARAM_CONFIG, jfile, indent=4)
    with open(f'{model_folder}/model_cfg.json', 'w') as jfile:
        json.dump(model_config, jfile, indent=4)

    LOGGER.info('Loading K TF datasets')

    # Create datasets for each fold
    ds_folds = []
    sample_count = 0
    for fold in range(1, n_folds + 1):
        dataset = load_and_process_dataset(dataset_folder, fold, batch_size)
        dataset_size = get_dataset_size(dataset, img_count=False)
        LOGGER.info(f'Loaded fold {fold} with {dataset_size} samples')
        sample_count += dataset_size
        ds_folds.append(dataset)
    LOGGER.info(f'Loaded {sample_count} samples in total')
    img_count = sample_count * batch_size
    LOGGER.info(
        f'With a batch size of {batch_size}, this corresponds to {img_count} images')

    # Prepare fold configurations
    fold_configurations = prepare_fold_configurations(n_folds, model_folder)
    models = []

    # Define strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()
    num_workers = strategy.num_replicas_in_sync
    devices = strategy.extended.worker_devices
    LOGGER.info(
        f"Number of devices (workers): {num_workers}\nDevices: {devices}")

    for i, (fold_entry) in enumerate(fold_configurations):
        if i < start_fold:
            LOGGER.info(f'Skipping fold {i + 1} as it was already completed.')
            continue

        # Set up training and validation datasets
        fold_val = fold_entry['validation_fold']
        folds_train = fold_entry['training_folds']
        fold_test = fold_entry['testing_fold']
        model_path = fold_entry['model_fold_path']

        ds_folds_train = [ds_folds[tid] for tid in folds_train]
        ds_train = reduce(tf.data.Dataset.concatenate, ds_folds_train)
        ds_val = ds_folds[fold_val]

        df_train_size = get_dataset_size(ds_train, img_count=True)
        df_val_size = get_dataset_size(ds_val, img_count=True)
        df_test_size = get_dataset_size(ds_folds[fold_test], img_count=True)
        LOGGER.info(
            f'Step {i + 1}/{n_folds} \n'
            f'Train folds {folds_train}\n\tsize: {df_train_size} images \n'
            f'Val fold: {fold_val}\n\tsize: {df_val_size} images \n'
            f'Test fold: {fold_test}\n\tsize: {df_test_size} images')

        if HYPER_PARAM_CONFIG['repeat_dataset'] is True:
            # repeat dataset only if not using all training data
            ds_train = ds_train.repeat()

        # Perform fitting using the strategy scope
        with strategy.scope():
            init_start = time.time()
            model = initialise_model(
                input_shape, model_config, chkpt_folder=chkpt_folder)
            callbacks = initialise_callbacks(model_path)
            init_end = time.time()
            init_duration = init_end - init_start
            LOGGER.info(f'\tTraining model, writing to {model_path}')

            # Fit model
            fitting_start_time = time.time()
            try:
                if HYPER_PARAM_CONFIG['repeat_dataset'] is True:
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
                                  callbacks=callbacks)
            except Exception as e:
                LOGGER.error(f'Error while fitting model: {e}')
                exit(1)
            fitting_end_time = time.time()
            fitting_duration = fitting_end_time - fitting_start_time

            # Append model to model list
            models.append(model)

            # Evaluate model on left-out fold
            LOGGER.info(f'Evaluating model on left-out fold {fold_test}')
            testing_start_time = time.time()
            evaluation = model.net.evaluate(ds_folds[fold_test])
            testing_end_time = time.time()
            testing_duration = testing_end_time - testing_start_time
            evaluation_dict = dict(zip(model.net.metrics_names, evaluation))
            evaluation_path = os.path.join(model_path, 'evaluation.json')
            with open(evaluation_path, 'w') as jfile:
                json.dump(evaluation_dict, jfile, indent=4)
            LOGGER.info(f'\tEvaluation results saved to {model_path}')

            # Registering fold configuration and training duration
            fold_duration = init_duration + fitting_duration + testing_duration
            model_size = callbacks[-1].model_size
            LOGGER.info(f'\n'
                        f'Fold {fold_test} completed in {fold_duration} seconds \n'
                        f'Model init duration: {init_duration} seconds \n'
                        f'Model fitting duration: {fitting_duration} seconds \n'
                        f'Model testing duration: {testing_duration} seconds \n')
            fold_infos = {
                'testing_fold': fold_test,
                'training_folds': folds_train,
                'validation_fold': fold_val,
                'model_size_gb': model_size,
                'fold_duration': fold_duration,
                'init_duration': init_duration,
                'fitting_duration': fitting_duration,
                'testing_duration': testing_duration
            }
            fold_data_path = os.path.join(model_path, 'fold_infos.json')
            with open(fold_data_path, 'w') as jfile:
                json.dump(fold_infos, jfile, indent=4)
        LOGGER.info(f'Fold {fold_test} completed')
    LOGGER.info('All folds completed')

    # Creating average model
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

    # Evaluate average model on all folds
    for fold_entry in fold_configurations:
        testing_id = fold_entry['testing_fold']
        LOGGER.info(
            f'Evaluating average model on left-out fold {testing_id}')
        avg_evaluation = avg_model.net.evaluate(ds_folds[testing_id])
        avg_evaluation_dict = dict(
            zip(avg_model.net.metrics_names, avg_evaluation))
        evaluation_path = os.path.join(model_path, 'evaluation_avg.json')
        with open(evaluation_path, 'w') as jfile:
            json.dump(avg_evaluation_dict, jfile, indent=4)
            LOGGER.info(
                f'\tEvaluation results for average model saved to {evaluation_path}')

    # Save training duration
    train_full_end_time = time.time()
    LOGGER.info(
        f'Training all models and average model took '
        f'{train_full_end_time - training_full_start_time} seconds')
    with open(f'{model_folder}/duration_seconds.txt', 'w') as txtfile:
        txtfile.write(str(train_full_end_time - training_full_start_time))


if __name__ == '__main__':
    if NIVA_PROJECT_DATA_ROOT is None:
        LOGGER.error('NIVA_PROJECT_DATA_ROOT environment variable not set')
        exit(1)
    if len(sys.argv) != 2:
        LOGGER.error('Usage: python training.py <model_name>')
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
    LOGGER.info('Training completed')

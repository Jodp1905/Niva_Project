import os
import sys
import json
import logging
import signal
from datetime import datetime
from functools import reduce, partial
from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import TanimotoDistanceLoss
from eoflow.models.segmentation_unets import ResUnetA

from tf_data_utils import (
    npz_dir_dataset,
    normalize_meanstd,
    Unpack, ToFloat32, augment_data, FillNaN, OneMinusEncoding, LabelsToDict)


# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Script parameters
NORMALIZER = dict(to_medianstd=partial(normalize_meanstd, subtract='median'))
TF_PROFILING = False
AUTOTUNE = tf.data.experimental.AUTOTUNE
UPDATE_FREQ = 'epoch'
PROFILE_BATCH = 0


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, update_freq=UPDATE_FREQ, profile_batch=PROFILE_BATCH):
        super().__init__(log_dir=log_dir, update_freq=update_freq,
                         profile_batch=profile_batch)
        self.profile_batch = profile_batch
        self.log_dir = log_dir
        self.tf_profiling = TF_PROFILING

    def on_epoch_begin(self, epoch, logs=None):
        if self.tf_profiling:
            profiler_logdir = f"{self.log_dir}/profiler_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tf.profiler.experimental.start(profiler_logdir)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.tf_profiling:
            tf.profiler.experimental.stop()
        self._writer.flush()


def get_dataset(npz_folder, metadata_path, fold, augment,
                augmentations_features, augmentations_label, num_parallel, randomize=True):

    data = dict(X='features', y_extent='y_extent',
                y_boundary='y_boundary', y_distance='y_distance')

    dataset = npz_dir_dataset(os.path.join(npz_folder, f'fold_{fold}'), data,
                              metadata_path=metadata_path, fold=fold, randomize=randomize,
                              num_parallel=num_parallel)

    normalizer = NORMALIZER["to_medianstd"]

    augmentations = [augment_data(
        augmentations_features, augmentations_label)] if augment else []
    dataset_ops = [normalizer, Unpack(), ToFloat32()] \
        + augmentations \
        + [FillNaN(fill_value=-2),
           OneMinusEncoding(
            n_classes=2),
           LabelsToDict(["extent", "boundary", "distance"])]

    for dataset_op in dataset_ops:
        dataset = dataset.map(dataset_op, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def initialise_model(input_shape, model_config, chkpt_folder=None):
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
    now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
    model_path = f'{model_folder}/{model_name}_fold-{fold}_{now}'

    os.makedirs(model_path, exist_ok=True)

    logs_path = os.path.join(model_path, 'logs')
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')

    tensorboard_callback = CustomTensorBoard(
        log_dir=logs_path, update_freq='epoch', profile_batch=0)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoints_path, save_best_only=True, save_freq='epoch', save_weights_only=True)

    with open(f'{model_path}/model_cfg.json', 'w') as jfile:
        json.dump(model_config, jfile)

    callbacks = [tensorboard_callback, checkpoint_callback]
    return model_path, callbacks


def train_k_folds(npz_folder, metadata_path, model_folder, chkpt_folder, input_shape,
                  n_classes, batch_size, iterations_per_epoch, num_epochs,
                  model_name, n_folds, seed, augmentations_features,
                  augmentations_label, model_config, wandb_id):

    if wandb_id is not None:
        os.system(f'wandb login {wandb_id}')

    LOGGER.info('Create K TF datasets')
    ds_folds = [get_dataset(npz_folder,
                            metadata_path,
                            fold=fold,
                            augment=True,
                            augmentations_features=augmentations_features,
                            augmentations_label=augmentations_label,
                            # TODO is num_parallel set to an optimal value?
                            num_parallel=100)
                for fold in tqdm(range(1, n_folds + 1))]

    folds = list(range(n_folds))
    folds_ids_list = [(folds[:nf] + folds[1 + nf:], [nf]) for nf in folds]
    np.random.seed(seed)

    models = []
    model_paths = []

    strategy = tf.distribute.MirroredStrategy()

    for training_ids, testing_id in folds_ids_list:
        left_out_fold = testing_id[0] + 1
        LOGGER.info(f'Training model for left-out fold {left_out_fold}')

        fold_val = np.random.choice(training_ids)
        folds_train = [tid for tid in training_ids if tid != fold_val]
        LOGGER.info(
            f'\tTrain folds {folds_train}, Val fold: {fold_val}, Test fold: {testing_id[0]}')

        ds_folds_train = [ds_folds[tid] for tid in folds_train]
        ds_train = reduce(tf.data.Dataset.concatenate, ds_folds_train)
        ds_val = ds_folds[fold_val].batch(batch_size)
        ds_train = ds_train.batch(batch_size).repeat()

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

    now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
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
        model.net.evaluate(ds_folds[testing_id[0]].batch(batch_size))
        LOGGER.info(
            f'Evaluating average model on left-out fold {left_out_fold}')
        avg_model.net.evaluate(ds_folds[testing_id[0]].batch(batch_size))
        LOGGER.info('\n\n')


if __name__ == '__main__':
    # Define paths
    NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
    NPZ_FILES_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/npz_files/')
    METADATA_PATH = Path(
        f'{NIVA_PROJECT_DATA_ROOT}/patchlets_dataframe_final.csv')
    KFOLD_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/folds/')
    MODEL_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/model/')
    CHKPT_FOLDER = None
    wandb_id = None
    input_shape = [256, 256, 4]
    n_classes = 2
    batch_size = 8
    iterations_per_epoch = 30
    num_epochs = 2
    model_name = "resunet-a"
    n_folds = 10
    seed = 42
    augmentations_features = ["flip_left_right",
                              "flip_up_down", "rotate", "brightness"]
    augmentations_label = ["flip_left_right", "flip_up_down", "rotate"]

    model_config = {
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

    train_k_folds(KFOLD_FOLDER, METADATA_PATH, MODEL_FOLDER, CHKPT_FOLDER,
                  input_shape, n_classes, batch_size, iterations_per_epoch,
                  num_epochs, model_name, n_folds, seed, augmentations_features,
                  augmentations_label, model_config, wandb_id)

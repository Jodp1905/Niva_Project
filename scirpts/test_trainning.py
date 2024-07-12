# Import des bibliothèques nécessaires
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from functools import reduce
from typing import List, Tuple, Callable, Union

import numpy as np
import tensorflow as tf
from fs.copy import copy_dir
from tqdm.auto import tqdm 

from wandb.keras import WandbCallback

from eoflow.models.metrics import MCCMetric
from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import JaccardDistanceLoss, TanimotoDistanceLoss
from eoflow.models.segmentation_unets import ResUnetA

# Initialisation du système de logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(logging.Filter("main"))
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

class Unpack(object):
    """ Unpack items of a dictionary to a tuple """
    def __call__(self, sample: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        return sample['features'], sample['labels']
    
class FillNaN(object):
    """ Replace NaN values with a given finite value """
    def __init__(self, fill_value: float = -2.):
        self.fill_value = fill_value
        
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        feats = tf.where(tf.math.is_nan(feats), tf.constant(self.fill_value, feats.dtype), feats)
        return feats, labels
    
class OneMinusEncoding(object):
    """ Encodes labels to 1-p, p. Makes sense only for binary labels and for continuous labels in [0, 1] """
    def __init__(self, n_classes: int):
        assert n_classes == 2, 'OneMinus works only for "binary" classes. `n_classes` should be 2.'
        self.n_classes = n_classes
      
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return feats, tf.concat([tf.ones_like(labels) - labels, labels], axis=-1)
    
def normalize_meanstd(ds_keys: dict, subtract: str = 'mean') -> dict:
    """ Help function to normalise the features by the mean and standard deviation """
    assert subtract in ['mean', 'median']
    feats = tf.math.subtract(tf.cast(ds_keys['features'], tf.float64), ds_keys[f'norm_meanstd_{subtract}'])
    feats = tf.math.divide(feats, ds_keys['norm_meanstd_std'])
    ds_keys['features'] = feats
    return ds_keys

def normalize_perc(ds_keys: dict) -> dict:
    """ Help function to normalise the features by the 99th percentile """
    feats = tf.math.divide(tf.cast(ds_keys['features'], tf.float64), ds_keys['norm_perc99'])
    ds_keys['features'] = feats
    return ds_keys

# Normaliseur
NORMALIZER = dict(to_meanstd=normalize_meanstd, to_perc=normalize_perc)

# Fonction pour normaliser les données
def normalize_data(dataset, normalize):
    if normalize in NORMALIZER:
        return NORMALIZER[normalize](dataset)
    else:
        raise ValueError(f"Normalize mode {normalize} not supported.")
    
def npz_dir_dataset(file_dir_or_list: Union[str, List[str]], features: dict, metadata_path: str,
                    fold: int = None, randomize: bool = True,
                    num_parallel: int = 5, shuffle_size: int = 500,
                    filesystem: S3FS = None, npz_from_s3: bool = False) -> tf.data.Dataset:
    """ Creates a tf.data.Dataset from a directory containing numpy .npz files.

    Files are loaded lazily when needed. `num_parallel` files are read in parallel and interleaved together.

    :param file_dir_or_list: directory containing .npz files or a list of paths to .npz files
    :param features: dict of (`field` -> `feature_name`) mappings, where `field` is the field in the .npz array
                   and `feature_name` is the name of the feature it is saved to.
    :param fold: in k-fold validation, fold to consider when querying the patchlet info dataframe
    :param randomize: whether to shuffle the samples of the dataset or not, defaults to `True`
    :param num_parallel: number of files to read in parallel and intereleave, defaults to 5
    :param shuffle_size: buffer size for shuffling file order, defaults to 500
    :param metadata_path: path to input csv files with patchlet information
    :param filesystem: filesystem to access bucket, defaults to None
    :param npz_from_s3: if True, npz files are loaded from S3 bucket, otherwise from local disk
    :return: dataset containing examples merged from files
    """

    files = file_dir_or_list

    if npz_from_s3:
        assert filesystem is not None
    
    # If dir, then list files
    if isinstance(file_dir_or_list, str):
        if filesystem and not filesystem.isdir(file_dir_or_list):
            filesystem.makedirs(file_dir_or_list)
        dir_list = os.listdir(file_dir_or_list) if not npz_from_s3 else filesystem.listdir(file_dir_or_list)
        files = [os.path.join(file_dir_or_list, f) for f in dir_list]
        
    fields = list(features.keys())

    # Read one file for shape info
    file = next(iter(files))
    data = np.load(file) if not npz_from_s3 else np.load(filesystem.openbin(file))
    np_arrays = [data[f] for f in fields]

    # Append norm arrays 
    perc99, meanstd_mean, meanstd_median, meanstd_std = _construct_norm_arrays(file, metadata_path, fold, filesystem)
    
    np_arrays.append(perc99)
    np_arrays.append(meanstd_mean)
    np_arrays.append(meanstd_median)
    np_arrays.append(meanstd_std)

    # Read shape and type info
#     types = tuple(arr.dtype for arr in np_arrays)
    types = (tf.uint16, tf.float32, tf.float32, tf.float32, tf.float64, tf.float64, tf.float64, tf.float64)
    shapes = tuple(arr.shape[1:] for arr in np_arrays)

    # Create datasets
    datasets = [_npz_file_lazy_dataset(file, fields, types, shapes, metadata_path, fold=fold,
                                       filesystem=filesystem, npz_from_s3=npz_from_s3) for file in files]
    ds = tf.data.Dataset.from_tensor_slices(datasets)

    # Shuffle files and interleave multiple files in parallel
    if randomize:
        ds = ds.shuffle(shuffle_size)
    
    ds = ds.interleave(lambda x: x, cycle_length=num_parallel)

    return ds

# Fonction pour préparer le dataset
def prepare_dataset(npz_folder, fold, metadata_path, augment=True, num_parallel=4, randomize=True, npz_from_s3=False):
    # Création du dossier pour le modèle si inexistant
    model_folder = os.path.join(npz_folder, f'fold_{fold}')
    os.makedirs(model_folder, exist_ok=True)

    # Normalisation des données
    dataset = npz_dir_dataset(model_folder, data=dict(X='features', y_extent='y_extent', y_boundary='y_boundary', y_distance='y_distance'),
                              metadata_path=metadata_path, fold=fold, randomize=randomize, num_parallel=num_parallel, npz_from_s3=npz_from_s3)

    dataset = normalize_data(dataset, normalize)

    augmentations = [augment_data(augmentations_feature, augmentations_label)] if augment else []
    dataset_ops = [Unpack(), ToFloat32()] + augmentations + [FillNaN(fill_value=-2),
                                                             OneMinusEncoding(n_classes=n_classes),
                                                             LabelsToDict(reference_names)]

    for dataset_op in dataset_ops:
        dataset = dataset.map(dataset_op)

    return dataset

# Fonction pour initialiser le modèle
def initialise_model(input_shape, n_classes, model_config, chkpt_folder=None):
    model = ResUnetA(model_config)
    model.build(dict(features=[None] + list(input_shape)))
    
    model.net.compile(
        loss={'extent': TanimotoDistanceLoss(from_logits=False),
              'boundary': TanimotoDistanceLoss(from_logits=False),
              'distance': TanimotoDistanceLoss(from_logits=False)},
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate']),
        metrics=[segmentation_metrics['accuracy'](),
                 tf.keras.metrics.MeanIoU(num_classes=n_classes)]
    )
    
    if chkpt_folder is not None:
        model.net.load_weights(f'{chkpt_folder}/model.ckpt')
    
    return model

# Fonction pour initialiser les callbacks
def initialise_callbacks(model_folder, fold):
    now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
    model_path = f'{model_folder}/{model_name}_fold-{fold}_{now}'

    os.makedirs(model_path, exist_ok=True)
    logs_path = os.path.join(model_path, 'logs')
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')

    # Callback pour TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path,
                                                          update_freq='epoch',
                                                          profile_batch=0)

    # Callback pour sauvegarde du modèle
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path,
                                                             save_best_only=True,
                                                             save_freq='epoch',
                                                             save_weights_only=True)

    # Configuration complète du modèle
    full_config = dict(**model_config,
                       iterations_per_epoch=iterations_per_epoch,
                       num_epochs=num_epochs,
                       batch_size=batch_size,
                       model_name=f'{model_name}_{now}')

    # Sauvegarde de la configuration du modèle
    with open(os.path.join(model_path, 'model_cfg.json'), 'w') as jfile:
        json.dump(model_config, jfile)

    # Initialisation de Weights & Biases si nécessaire
    if wandb_id:
        wandb.init(config=full_config,
                   name=f'{model_name}-leftoutfold-{fold}',
                   project='field-delineation',
                   sync_tensorboard=True)

    # Liste des callbacks
    callbacks = [tensorboard_callback,
                 checkpoint_callback,
                 WandbCallback()] if wandb_id else [tensorboard_callback, checkpoint_callback]

    return model_path, callbacks

# Fonction principale pour l'entraînement en k-fold
def train_k_folds(npz_folder, metadata_path, model_folder, model_s3_folder, input_shape, n_classes, batch_size,
                  iterations_per_epoch, num_epochs, model_name, augmentations_feature, augmentations_label,
                  reference_names, normalize, model_config, chkpt_folder=None, n_folds=5, wandb_id=None):
    
    LOGGER.info('Création des ensembles de données TF pour chaque fold')

    # Boucle sur chaque fold
    for fold in tqdm(range(1, n_folds + 1), desc='Folds'):
        LOGGER.info(f'Entraînement sur le fold {fold}')

        # Préparation du dataset
        dataset = prepare_dataset(npz_folder, fold, metadata_path, augment=True, num_parallel=4, randomize=True, npz_from_s3=False)

        # Initialisation du modèle
        model = initialise_model(input_shape, n_classes, chkpt_folder)

        # Initialisation des callbacks
        model_path, callbacks = initialise_callbacks(model_folder, fold)

        # Entraînement du modèle
        model.net.fit(dataset.batch(batch_size),
                      epochs=num_epochs,
                      steps_per_epoch=iterations_per_epoch,
                      callbacks=callbacks)

        # Libération de la mémoire
        del model, dataset

    LOGGER.info('Entraînement terminé avec succès')

    # Copies des répertoires de modèle dans le bucket S3
    for fold in tqdm(range(1, n_folds + 1), desc='Copie des modèles dans S3'):
        model_name_fold = f'{model_name}_fold-{fold}'
        model_path_fold = os.path.join(model_folder, model_name_fold)

        os.makedirs(os.path.join(model_s3_folder, model_name_fold), exist_ok=True)
        copy_dir(model_path_fold, os.path.join(model_s3_folder, model_name_fold))

    LOGGER.info('Copie des modèles dans S3 terminée')

    # Création du modèle moyen
    models = [initialise_model(input_shape, n_classes, chkpt_folder) for _ in range(n_folds)]
    weights = [model.net.get_weights() for model in models]

    avg_weights = [np.mean(np.array(w), axis=0) for w in zip(*weights)]

    avg_model = initialise_model(input_shape, n_classes, chkpt_folder)
    avg_model.net.set_weights(avg_weights)

    now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
    model_path_avg = os.path.join(model_folder, f'{model_name}_avg_{now}')

    LOGGER.info('Sauvegarde du modèle moyen en local')
    os.makedirs(model_path_avg, exist_ok=True)

    checkpoints_path_avg = os.path.join(model_path_avg, 'checkpoints', 'model.ckpt')
    with open(os.path.join(model_path_avg, 'model_cfg.json'), 'w') as jfile:
        json.dump(model_config, jfile)
    avg_model.net.save_weights(checkpoints_path_avg)

    LOGGER.info('Évaluation des modèles et du modèle moyen')

    # Boucle sur chaque fold pour évaluation
    for fold in tqdm(range(1, n_folds + 1), desc='Évaluation sur les folds'):
        LOGGER.info(f'Évaluation sur le fold {fold}')

        dataset_fold = prepare_dataset(npz_folder, fold, metadata_path, augment=False, num_parallel=4, randomize=False, npz_from_s3=False)

        LOGGER.info(f'Évaluation du modèle sur le fold {fold}')
        models[fold-1].net.evaluate(dataset_fold.batch(batch_size))

        LOGGER.info(f'Évaluation du modèle moyen sur le fold {fold}')
        avg_model.net.evaluate(dataset_fold.batch(batch_size))

        LOGGER.info('\n\n')

# Point d'entrée principal
if __name__ == '__main__':
    # Arguments en dur (à remplacer par vos valeurs spécifiques)
    npz_folder = "chemin/vers/votre/dossier/npz"
    metadata_path = "chemin/vers/votre/fichier/metadata.json"
    model_folder = "chemin/vers/votre/dossier/models"
    model_s3_folder = "chemin/vers/votre/bucket/s3/models"
    input_shape = (256, 256, 3)
    n_classes = 3
    batch_size = 8
    iterations_per_epoch = 1000
    num_epochs = 10
    model_name = "nom_de_votre_modele"
    augmentations_feature = True
    augmentations_label = True
    reference_names = ['extent', 'boundary', 'distance']
    normalize = 'to_meanstd'
    model_config = {
        'learning_rate': 0.001,
        'filters': 64,
        'num_res_blocks': 4
    }
    chkpt_folder = None  # Chemin vers un dossier de points de contrôle, si nécessaire
    n_folds = 5
    wandb_id = None  # ID de Weights & Biases, si utilisé

    # Appel à la fonction principale d'entraînement en k-fold avec les paramètres en dur
    train_k_folds(npz_folder=npz_folder,
                  metadata_path=metadata_path,
                  model_folder=model_folder,
                  model_s3_folder=model_s3_folder,
                  input_shape=input_shape,
                  n_classes=n_classes,
                  batch_size=batch_size,
                  iterations_per_epoch=iterations_per_epoch,
                  num_epochs=num_epochs,
                  model_name=model_name,
                  augmentations_feature=augmentations_feature,
                  augmentations_label=augmentations_label,
                  reference_names=reference_names,
                  normalize=normalize,
                  model_config=model_config,
                  chkpt_folder=chkpt_folder,
                  n_folds=n_folds,
                  wandb_id=wandb_id)

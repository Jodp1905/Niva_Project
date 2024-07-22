import sys
import os
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

# Configuration du logger
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

def multiprocess(process_fun, arguments, total=None, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_fun, arguments), total=total))
    return results

def fold_split(chunk: str, df: pd.DataFrame, npz_folder: str, folds_folder: str, n_folds: int):
    # Créer les dossiers pour chaque fold s'ils n'existent pas déjà
    for fold in range(1, n_folds + 1):
        fold_folder = os.path.join(folds_folder, f'fold_{fold}')
        os.makedirs(fold_folder, exist_ok=True)

    # Charger les données du fichier .npz
    data = np.load(os.path.join(npz_folder, chunk), allow_pickle=True)

    # Parcourir chaque fold pour extraire et sauvegarder les patchlets
    for fold in range(1, n_folds + 1):
        # Sélectionner les indices correspondant au fold actuel
        idx_fold = df[(df.chunk == chunk) & (df.fold == fold)].chunk_pos
        if not idx_fold.empty:
            # Créer un dictionnaire des patchlets pour le fold
            patchlets = {key: data[key][idx_fold] for key in data}
            fold_folder = os.path.join(folds_folder, f'fold_{fold}')
            # Sauvegarder les patchlets dans un fichier .npz
            np.savez(os.path.join(fold_folder, chunk), **patchlets)

def k_fold_split(metadata_path: str, npz_folder: str, n_folds: int, seed: int, folds_folder: str, max_workers: int = 4):
    LOGGER.info(f'Read metadata file {metadata_path}')
    df = pd.read_csv(metadata_path)
    eops = df.eopatch.unique()

    LOGGER.info('Assign folds to eopatches')
    np.random.seed(seed)
    fold = np.random.randint(1, high=n_folds + 1, size=len(eops))
    eopatch_to_fold_map = dict(zip(eops, fold))
    df['fold'] = df['eopatch'].apply(lambda x: eopatch_to_fold_map[x])

    for nf in range(n_folds):
        LOGGER.info(f'{len(df[df.fold == nf + 1])} patchlets in fold {nf + 1}')

    LOGGER.info('Split files into folds')
    partial_fn = partial(fold_split, df=df, npz_folder=npz_folder, folds_folder=folds_folder, n_folds=n_folds)
    npz_files = [npzf for npzf in os.listdir(npz_folder) if npzf.startswith('patchlets_')]
    _ = multiprocess(partial_fn, npz_files, max_workers=max_workers)

    LOGGER.info('Update metadata file with fold information')
    df.to_csv(metadata_path, index=False)

def k_folds(metadata_path: str, n_folds: int, seed: int, max_workers: int = 4):
    # Créer le dossier 'folds' à côté du fichier de métadonnées
    npz_folder = os.path.join(os.path.dirname(metadata_path), 'npz_files')
    folds_folder = os.path.join(os.path.dirname(metadata_path), 'folds')
    os.makedirs(folds_folder, exist_ok=True)
    k_fold_split(metadata_path, npz_folder, n_folds, seed, folds_folder, max_workers)

if __name__ == '__main__':
    # Définition des paramètres
    metadata_path = '/home/joseph/Code/localstorage/dataframe.csv'
    n_folds = 3
    seed = 2

    # Appel à la fonction k_folds
    k_folds(metadata_path, n_folds, seed)

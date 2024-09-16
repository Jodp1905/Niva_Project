import pandas as pd
import numpy as np
import os
import aiohttp
import aiofiles
import asyncio
from tqdm.asyncio import tqdm
from pathlib import Path
import logging
from filter import LogFileFilter
import sys
import urllib.error
import urllib.request
import shutil
import time

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Define paths
# local
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
SENTINEL2_DIR = Path(f'{NIVA_PROJECT_DATA_ROOT}/sentinel2/')
# web
AI4BOUNDARIES_URL = 'http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES'
AI4BOUNDARIES_SPLIT_FILE = 'ai4boundaries_ftp_urls_sentinel2_split.csv'
RATE_LIMIT = 5  # requests per second
RETRY_LIMIT = 3  # number of retries


async def download_file(session, url, dst_path):
    """
    Asynchronously download a single file to disk.

    :param session: aiohttp ClientSession
    :param url: URL of the file to download
    :param dst_path: File location on disk after download
    """
    try:
        async with session.get(url) as response:
            if response.status == 200:
                async with aiofiles.open(dst_path, mode='wb') as f:
                    await f.write(await response.read())
            else:
                LOGGER.error(f"Error {response.status} downloading {url}")
    except Exception as e:
        LOGGER.error(f"Exception occurred during download {url}: {e}")


async def help_func(session, data, path):
    """
    Asynchronous file download helper function.

    :param session: aiohttp ClientSession
    :param data: List of URLs to download
    :param path: Destination directory
    """
    fail_fns = []
    for url_fn in tqdm(data):
        fn = os.path.join(path, url_fn.rsplit("/", 1)[-1])
        try:
            await download_file(session, url_fn, fn)
        except Exception as e:
            LOGGER.error(f"Failed to download {url_fn}: {e}")
            fail_fns.append(url_fn)
            await asyncio.sleep(20)
    return fail_fns


async def download_images(fold_data, folder_save):
    """
    Asynchronously download images and masks from the dataset.

    :param fold_data: DataFrame with file URLs
    :param folder_save: Destination folder for downloads
    """
    async with aiohttp.ClientSession() as session:
        num_files = len(fold_data)

        # Download images
        LOGGER.info("Downloading sentinel2 images...")
        fail_fns_img = await help_func(
            session, fold_data["sentinel2_images_file_url"], path=os.path.join(
                folder_save, "images")
        )
        download_count_imgs = num_files - len(fail_fns_img)
        LOGGER.info(f"Downloaded {download_count_imgs} files to images folder\n"
                    f"failed files: {fail_fns_img}")
        for i in range(RETRY_LIMIT):
            if len(fail_fns_img) == 0:
                break
            LOGGER.info(f"Retrying failed downloads, attempt {i + 1}")
            await asyncio.sleep(2 ** i)
            fold_data_retry = fold_data[fold_data["sentinel2_images_file_url"].isin(
                fail_fns_img)]
            fail_fns_img = await help_func(
                session,
                fold_data_retry,
                path=os.path.join(folder_save, "masks")
            )

        # Download masks
        # sentinel2_images_file_url
        LOGGER.info("Downloading sentinel2 masks...")
        fail_fns_mask = await help_func(
            session, fold_data["sentinel2_masks_file_url"], path=os.path.join(
                folder_save, "masks")
        )
        download_count_masks = num_files - len(fail_fns_mask)
        LOGGER.info(f"Downloaded {download_count_masks} files to masks folder\n"
                    f"failed files: {fail_fns_mask}")
        for i in range(RETRY_LIMIT):
            if len(fail_fns_mask) == 0:
                break
            LOGGER.info(f"Retrying failed downloads, attempt {i + 1}")
            fold_data_retry = fold_data[fold_data["sentinel2_masks_file_url"].isin(
                fail_fns_mask)]
            fail_fns_mask = await help_func(
                session,
                fold_data_retry,
                path=os.path.join(folder_save, "masks")
            )


def random_train_set(data, country='SI', percentage=0.7, split='test'):
    """"
    Enrich training set by randomly setting validation/testing to training set
    """
    sub_data = data[(data['Country'] == country) & (data['split'] == split)]
    size_ = int(len(sub_data) * percentage)
    sub_ind = np.random.choice(sub_data.index, size_, replace=False)
    data.loc[sub_ind, 'new_split'] = 'train'
    return data


if __name__ == "__main__":
    if not NIVA_PROJECT_DATA_ROOT:
        raise ValueError(
            "NIVA_PROJECT_DATA_ROOT environment variable is not set")
    main_time_start = time.time()
    # connection check
    try:
        urllib.request.urlopen(AI4BOUNDARIES_URL)
    except urllib.error.URLError as e:
        LOGGER.error(f"Error connecting to {AI4BOUNDARIES_URL}: {e}")
        exit(1)

    split_filepath = os.path.join(SENTINEL2_DIR, AI4BOUNDARIES_SPLIT_FILE)
    os.makedirs(NIVA_PROJECT_DATA_ROOT, exist_ok=True)
    os.makedirs(SENTINEL2_DIR, exist_ok=True)

    # save split file from AI4boundaries dataset to local path_to_save
    try:
        urllib.request.urlretrieve(
            f"{AI4BOUNDARIES_URL}/sentinel2/{AI4BOUNDARIES_SPLIT_FILE}", split_filepath)
        LOGGER.info(
            f"Downloaded {AI4BOUNDARIES_SPLIT_FILE} to {split_filepath}")
    except urllib.error.URLError as e:
        LOGGER.error(f"Error downloading {AI4BOUNDARIES_SPLIT_FILE}: {e}")
        exit(1)

    data = pd.read_csv(split_filepath)
    data['Country'] = data['file_id'].str[:2]
    LOGGER.info(data.groupby(['Country', "split"],
                dropna=False)[['file_id']].count())
    data['new_split'] = data['split'].copy()

    # without split add to training set
    data.loc[data["split"].isna(), "new_split"] = "train"
    # all France scene in test split only
    data.loc[data["Country"] == "FR", "new_split"] = "test"
    # part of testing split add to training
    for country in ['NL', 'AT', 'SE', 'LU', 'ES', 'SI']:
        random_train_set(data, country=country, percentage=0.9, split='test')
    LOGGER.info(data.groupby(['Country', "new_split"],
                dropna=False)[['file_id']].count())

    # part of validation split add to training
    for country in ['NL', 'AT', 'SE', 'LU', 'ES', 'SI']:
        random_train_set(data, country=country, percentage=0.5, split='val')
    LOGGER.info(data.groupby(['Country', "new_split"],
                dropna=False)[['file_id']].count())
    LOGGER.info(data.groupby(["new_split"], dropna=False)[['file_id']].count())
    """
    test          2164
    train         5236
    val            431
    """
    data.to_csv(split_filepath, index=False)
    LOGGER.info(f"Saved updated split file to {split_filepath}")
    for fold in ["train", "val", "test"]:
        time_start = time.time()
        LOGGER.info(f"Downloading {fold} split")
        folder_save = os.path.join(SENTINEL2_DIR, fold)
        if os.path.exists(folder_save):
            LOGGER.info(
                f"Detected existing folder {folder_save} for {fold} split, cleaning up")
            shutil.rmtree(folder_save)

        os.makedirs(folder_save, exist_ok=True)
        os.makedirs(os.path.join(folder_save, "masks"), exist_ok=True)
        os.makedirs(os.path.join(folder_save, "images"), exist_ok=True)

        fold_data = data[data['new_split'] == fold]
        asyncio.run(download_images(fold_data, folder_save))
        time_end = time.time()
        dl_time_str = time.strftime(
            "%H:%M:%S", time.gmtime(time_end - time_start))
        LOGGER.info(
            f"Downloaded {fold} split to {folder_save} in {dl_time_str}")
    main_time_end = time.time()
    main_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(main_time_end - main_time_start))
    LOGGER.info(f"All splits downloaded in {main_time_str}")

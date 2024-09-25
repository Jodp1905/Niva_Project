# Environment Variable Overrides for Configuration

This document describes how to override configuration values from the YAML file using environment variables.

Each key in the YAML configuration has an equivalent environment variable name in uppercase.

To override any configuration parameter, you can set the corresponding environment variable before running your scripts.  
Some parameters expect specific types (e.g., integers, booleans), so ensure the values are set appropriately.

## How to Set Environment Variables

You can set environment variables in your shell using the `export` command (for Unix-like systems) or through your IDE’s environment variable settings.

Example:

```bash
export NIVA_PROJECT_DATA_ROOT="/path/to/data_root"
export NUM_EPOCHS=30
```

## General configuration

You should set NIVA_PROJECT_DATA_ROOT to the path of your choice before running any other script.
The provided path will be used to download datasets, preprocess them and save training model results.

| Environment Variable          | Description                             | Type    | Default Value                             |
|-------------------------------|-----------------------------------------|---------|-------------------------------------------|
| `NIVA_PROJECT_DATA_ROOT`      | Path to the project's data root folder  | `str`   | `null`                                    |

## Preprocessing

The preprocessing pipeline is composed of a few configurable steps with parameters exposed in this section.

### Download Data

| Environment Variable            | Description                                        | Type    | Default Value                             |
|---------------------------------|----------------------------------------------------|---------|-------------------------------------------|
| `AI4BOUNDARIES_URL`             | ftp server URL for downloading AI4BOUNDARIES data  | `str`   | `http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES` |
| `AI4BOUNDARIES_SPLIT_TABLE`     | Path to the file with split table for Sentinel-2   | `str`   | `ai4boundaries_ftp_urls_sentinel2_split.csv` |
| `DL_RATE_LIMIT`                 | Rate limit for downloads in requests per second    | `int`   | `5`                                          |
| `DL_RETRY_LIMIT`                | Retry limit count for failed downloads             | `int`   | `3`                                          |
| `FRACTION_DOWNLOAD`             | Fraction of the dataset to download                | `float` | `1.0`                                        |

### Create EOPatches

No modifiable parameters

### Create NPZ

| Environment Variable          | Description                                               | Type    | Default Value                             |
|-------------------------------|-----------------------------------------------------------|---------|-------------------------------------------|
| `NPZ_CHUNK_SIZE`              | Number of eopatches data concatenated in each NPZ chunk   | `int`   | `20`                                      |

### Create Datasets

| Environment Variable            | Description                                              | Type    | Default Value   |
|---------------------------------|----------------------------------------------------------|---------|-----------------|
| `SHUFFLE_BUFFER_SIZE`           | Size of the shuffle buffer (tf shuffle function)         | `int`   | `2000`          |
| `INTERLEAVE_CYCLE_LENGTH`       | Number of interleave cycles (tf interleave function)     | `int`   | `10`            |
| `ENABLE_AUGMENTATION`           | Whether to enable data augmentation                      | `bool`  | `true`          |
| `USE_FILE_SHARDING`             | Whether to use file sharding                             | `bool`  | `false`         |
| `NUM_SHARDS`                    | Number of shards to create if fale sharding is activated | `int`   | `35`            |

## Model Configuration

list type should be set with environment variables as a comma separated string.

Example:

```bash
export DILATION_RATE="1,3,5,8"
```

| Environment Variable        | Description                                        | Type    | Default Value                             |
|-----------------------------|----------------------------------------------------|---------|-------------------------------------------|
| `MODEL_NAME`                | Name of the model                                  | `str`   | `resunet-a`                               |
| `INPUT_SHAPE`               | Input shape for the model                          | `list`  | `[256, 256, 4]`                           |
| `LEARNING_RATE`             | Learning rate for the model                        | `float` | `0.0001`                                  |
| `N_LAYERS`                  | Number of layers in the model                      | `int`   | `3`                                       |
| `N_CLASSES`                 | Number of classes to predict                       | `int`   | `2`                                       |
| `KEEP_PROB`                 | Keep probability for dropout layers                | `float` | `0.8`                                     |
| `FEATURES_ROOT`             | Number of features at the root layer               | `int`   | `32`                                      |
| `CONV_SIZE`                 | Size of convolution filters                        | `int`   | `3`                                       |
| `CONV_STRIDE`               | Stride of convolutions                             | `int`   | `1`                                       |
| `DILATION_RATE`             | Dilation rate for convolutions                     | `list`  | `[1, 3, 15, 31]`                          |
| `DECONV_SIZE`               | Size of deconvolution filters                      | `int`   | `2`                                       |
| `ADD_DROPOUT`               | Whether to add dropout layers                      | `bool`  | `true`                                    |
| `ADD_BATCH_NORM`            | Whether to add batch normalization                 | `bool`  | `false`                                   |
| `USE_BIAS`                  | Whether to use bias in the layers                  | `bool`  | `false`                                   |
| `BIAS_INIT`                 | Initial value for bias                             | `float` | `0.0`                                     |
| `PADDING`                   | Padding method for convolutions                    | `str`   | `SAME`                                    |
| `POOL_SIZE`                 | Size of pooling layers                             | `int`   | `3`                                       |
| `POOL_STRIDE`               | Stride for pooling layers                          | `int`   | `2`                                       |
| `PREDICTION_VISUALIZATION`  | Whether to visualize predictions                   | `bool`  | `true`                                    |
| `CLASS_WEIGHTS`             | Class weights for weighted loss function           | `list`  | `null`                                    |

## Training

| Environment Variable        | Description                                        | Type    | Default Value                             |
|-----------------------------|----------------------------------------------------|---------|-------------------------------------------|
| `NUM_EPOCHS`                | Number of training epochs                          | `int`   | `20`                                      |
| `BATCH_SIZE`                | Batch size for training                            | `int`   | `8`                                       |
| `ITERATIONS_PER_EPOCH`      | Number of iterations per epoch                     | `int`   | `-1` (all data)                           |
| `TRAINING_TYPE`             | Type of training (SingleWorker or MultiWorker)     | `str`   | `SingleWorker`                            |
| `USE_NPZ`                   | Whether to use NPZ files for training              | `bool`  | `false`                                   |
| `TF_FULL_PROFILING`         | Enable full TensorFlow profiling                   | `bool`  | `false`                                   |
| `PREFETCH_DATA`             | Whether to prefetch data during training           | `bool`  | `true`                                    |
| `ENABLE_DATA_SHARDING`      | Whether to enable data sharding during training    | `bool`  | `true`                                    |
| `CHKPT_FOLDER`              | Folder to store checkpoints                        | `str`   | `null`                                    |
| `TENSORBOARD_UPDATE_FREQ`   | Frequency of updates for TensorBoard               | `str`   | `epoch`                                   |

## Inference

### download tile

| Environment Variable        | Description                                                  | Type    | Default Value                   |
|-----------------------------|--------------------------------------------------------------|---------|---------------------------------|
| `INPUT_COORDS_PATH`         | Path to the GeoJson file used as inference input             | `str`   | `null`                          |
| `TILE_NAME`                 | Name of the created NetCDF4 tile file                        | `str`   | `null`                          |
| `XARRAY_CHUNK_SIZE`         | Size of a chunk for Dask parallel processing of input tile   | `int`   | `2048`                          |

### create subtiles

| Environment Variable        | Description                                                  | Type    | Default Value                   |
|-----------------------------|--------------------------------------------------------------|---------|---------------------------------|
| `HEIGHT`                    | Height of an eopatch generated from NetCDF4 file tile        | `int`   | `1000`                          |
| `WIDTH`                     | Width of an eopatch generated from NetCDF4 file tile         | `int`   | `1000`                          |
| `OVERLAP`                   | (NOT IMPLEMENTD) Overlap value between consecutive generated eopacthes | `int`   | `0`                   |
| `NUM_SPLIT`                 | Limit number of eopatches to generate                        | `int`   | `-1` (no limit)                 |
| `BEGIN_X`                   | x-axis coordinate where eopatch sampling starts              | `int`   | `0`                             |
| `BEGIN_Y`                   | y-axis coordinate where eopatch sampling starts              | `int`   | `0`                             |

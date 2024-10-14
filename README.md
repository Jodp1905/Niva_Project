# Niva Project

Field delineation using the Resunet-a model.

## About The Project

This projects is based on the sentine-hub [field delineation project](https://github.com/sentinel-hub/field-delineation).

## Getting Started

This section will guide you on setting up the project and executing the full deep learning training and inference pipeline

### Prerequisites

#### Python

This project uses python 3.10, that can be installed alongside other versions.

* Add python repository

  ```sh
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  ```

* Install python 3.10

  ```sh
  sudo apt install python3.10 python3.10-venv python3.10-dev
  python3.10 --version
  ```

You may also want to install it from source by downloading the tarball from
[the official python website](https://www.python.org/downloads/release/python-31015/),
extracting it and following instructions.

#### GEOS

The [geos library](https://github.com/libgeos/geos) is required by python modules used.

* Install with apt

  ```sh
  sudo apt update
  sudo apt-get install libgeos-dev
  ```

* to install from source, see <https://github.com/libgeos/geos/blob/main/INSTALL.md>

#### PostgreSQL (INFERENCE ONLY)

libpq is the C application programmer's interface to PostgreSQL. It is required to install
the `psycopg2` package

* Install with apt

  ```sh
  sudo apt update
  sudo apt-get install libpq-dev
  ```

* to install from source, see <https://www.postgresql.org/docs/current/installation.html>

#### To use profiling tools

The project comes with a script allowing the use of the [Darshan I/O profiler](https://docs.nersc.gov/tools/performance/darshan/) or the [Nsight system performance analysis tool](https://developer.nvidia.com/nsight-systems) during training and data preprocessing. Both should be installed to generate traces.

Both Darshan and Nsight comes bundled with tools separated into host and target sections:

* target tools should be installed on the compute node executing the training program and is responsible for generating raw traces that are not directly readable.
* host tools should be installed on your personnal workstation (with a screen) and are used to visualize and analyze tracing results.

Thus, if you work in a HPC environment, Darshan and Nsight should be installed on the system you use for computations and on the one you use for analysis.

* Nsight can be downloaded from [here](https://developer.nvidia.com/nsight-systems/get-started)
  
  Extensive documentation: [Get Started With Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started)

* Darshan can be downloaded from [here](https://www.mcs.anl.gov/research/projects/darshan/download/)
  
  See [Darshan runtime installation and usage](https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-runtime.html) to install darshan-runtime on target computer.

  See [Darshan-util installation and usage](https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html) to install darshan-utils on host computer.

  The python Darshan package is the most convenient way to create summaries from binary trace files. It is already included in the virtual environnment set up at the next step.

### Installation

#### 1. Clone the repository

```sh
git clone https://github.com/Jodp1905/Niva_Project.git
```

#### 2. Configure the project

You can modify some parameters used in the project in the `config.yaml` file under `/config`.

YAML config file values can also be overriden by environment variables if set,
see `ENV_CONFIG.md` under `/config` for a description of all parameters.

Upon downloading the project, all parameters will first be set to their default values.

⚠️ Warning: **niva_project_data_root** must be set for any part of the project to run.
All data used for training, the trained models, and the inference results will be written
in this directory.

Set it in the yaml file where it is null by default, or export it as an environment variable:

```sh
mkdir /path/to/project_data/
export NIVA_PROJECT_DATA_ROOT=/path/to/project_data/
```

You may also add the line to your ~/.bashrc file for convenience.

#### 3. Install required python packages

You can download the necessary python packages by using the requirements files at
the root of the project:

* **`requirements_training.txt`** contains necessary packages for running the training
* **`requirements_inference.txt`** contains necessary packages for running the inference pipeline

They are separated to allow for more flexibility in the install process as the inference pipeline
requires a PostgreSQL install.

The use of a virtual environment is advised:

* Create and activate virtual environment

  ```sh
  python3.10 -m venv /path/to/niva-venv
  source /path/to/niva-venv/bin/activate
  ```

* Install from requirements.txt in the venv

  ```sh
  pip install -r requirements.txt
  ```

### Usage

This sections provides an overview on how you can get started and run the full field
delineation training pipeline from end to end. Bash scripts are made available under `/scripts`
to facilitate executions, but you may also use the python scripts under `/src` directly.

#### 1. Download dataset

To download the ai4boundaries dataset from the Joint Research Centre Data Catalogue
ftp servers, use the `download_dataset.sh` script:

```sh
cd ./scripts
./download_dataset.sh
```

Data is downloaded at the location specified by **niva_project_data_root** under `/sentinel2`
and split into 3 folders corresponding to the training configurations:
**training/validation/testing**. Requires an internet connection.

You should have this structure after download :

```txt
niva_project_data_root
└── sentinel2
    ├── ai4boundaries_ftp_urls_sentinel2_split.csv
    ├── test
    │   ├── images        # 2164 files
    │   └── masks         # 2164 files
    ├── train
    │   ├── images        # 5236 files
    │   └── masks         # 5236 files
    └── val
        ├── images        # 431 files
        └── masks         # 432 files
```

#### 2. Preprocessing pipeline

The downloaded dataset can now be preprocessed using the `run_preprocessing.sh` script:

```sh
cd ./scripts
./run_preprocessing.sh
```

The preprocessing pipeline create multiple folders under **niva_project_data_root**
corresponding to its different executed steps while keeping the test/train/val structure :

```txt
niva_project_data_root
├── datasets
│   ├── test
│   ├── train
│   └── val
├── eopatches
│   ├── test
│   ├── train
│   └── val
├── npz_files
│   ├── test
│   ├── train
│   └── val
├── patchlets_dataframe.csv
└── sentinel2
```

#### 3. Training

Once the datasets are created, you can run the model training using the `training.sh` script :

```sh
cd ./scripts
./run_training.sh
```

After training, the resulting model will be saved under `/model` in **niva_project_data_root**
as "training_$date" by default:

```txt
niva_project_data_root
├── datasets
├── eopatches
├── sentinel2
├── npz_files
├── patchlets_dataframe.csv
└── models
    ├── training_20240922_160621
    └── training_20240910_031256
```

You may also directly execute the python script and set a name of your choice :

```sh
cd ./src/training
python3 training.py <training-name>
```

Once the training has been executed, you can use the training name and the `main_analyze`
script under /utils to generate loss/accuracy plots as well as a textual description of
hyperparameters, memory usage or model size:

```sh
cd ./src/training
python3 main_analyze.py <training-name>
```

#### 4. Tracing

**Generating traces**

You will first have to set the following parameters in `tracing_wrapper.sh` under `/script`:

* PYTHON_VENV_PATH: The path to the root folder of your python environment used to run training.
* DARSHAN_LIBPATH: Path to `libdarshan.so`, should be under `path/to/darshan-runtime-install/lib/`.
* NSIGHT_LOGDIR: set tracefile output directory for nsight traces.

**Note**: The Darshan trace files will be saved under the path set upon installing darshan-runtime, use `darshan-config --log-path` to view it.

* DARSHAN_DXT: Set to 1 to enable [Darshan eXtended Tracing](https://docs.nersc.gov/tools/performance/darshan/dxt/), which generates more detailed I/O traces including a per file summary of all I/O operations.
* NSIGHT_NVTX: Training specific option. Set to 1 to enable profiling of samples of training batches instead of the entire training execution.
Note that `nsight_batch_profiling` also needs to be enabled in yaml configuration file.
* NSIGHT_STORAGE_METRICS: Set to 1 to enable using the storage_metrics nsight plugin, which should be included in you nsight distribution.
* LUSTRE_LLITE_DIR: Used with NSIGHT_STORAGE_METRICS=1, path to the lustre LLITE directory for capturing Lustre I/O metrics.
* NSIGHT_PYTHON_SAMPLING:  Set to 1 to enable nsight collection of Python backtrace sampling events. Overhead can be high

Once set, use the script to generate darshan or nsight reports of the training process:

``` sh
./tracing_wrapper.sh <darshan/nsight> ./run_training.sh 
```

Note that, as a wrapper, it can be user with any command/binary/script.
You can therefore use it directly with the python interpreter:

``` sh
./tracing_wrapper.sh <darshan/nsight> python3 ./../src/training/training.py
```

**Visualizing traces** 

Once the wrapped program has terminated, darshan (.darshan) or Nsight (.nsys-rep) files are created under the default log path for darshan or the one you set in the script parameters for Nsight. You may have to download them on your host machine to use the visualization tools.

You can use the following:

* Darshan:
  You can firstly use the darshan python package to create a job summary with heatmaps and I/O metrics for all enable darshan modules:
  
  ``` sh
  python3 -m darshan summary ./my_darshan_trace_file.darshan
  ```

  This will result in the creation of an html file, that you can view on a local live server.

  If you enabled DxT traces, you can use the dxt-parser command from darshan-utils to create a detailled report of timestamped I/O accesses for all files:
  
  ``` sh
  darshan-parser ./my_darshan_trace_file.darshan
  ```

* Nsight:
  
  Visualizing Nsight traces amounts to using the nsys-ui command:

  ``` sh
  nsys-ui ./my_nsys_trace_file.nsys-rep
  ```

## Inference pipeline

To run inference pipeline model (ResUnet-a) pretrained weights (and config) are REQUIRED!
In `config/config.cfg` it's parameters

```txt
prediction_config:
  model_path: "checkpoints"  # paste your own path to model checkpoint !!!!
  model_cfg_path: "model_cfg.json" # paste your own path to model configs !!!!
```

Other required parameters (niva_project_data_root_inf, TILE_ID):
```txt
  # INFERENCE
  niva_project_data_root_inf: "inference" # TO SET
  # TILE_ID: "S2A_30UWU_20230302_0_L2A"  # Tile in région Bretagne
  TILE_ID: "S2B_31TEN_20230420_0_L2A"  # Paste your own tile id to process
```
TILE_ID could be searched with the help of `scripts/notebooks/tile_search.ipynd`.

The command to download tile the chosen tile:
``` sh
  python src/inference/tile_download_by_id.py
```
The command to run inference on the downloaded tile:
``` sh
  python src/inference/main_inference.py
```
The output of the commands have the following structure:

```txt
[niva_project_data_root_inf]
  [TILE_ID]
    contours
    eopatches
    predictions
    tile
      [TILE_ID].nc
      metadata.gpkg
```
The final GeoJson file with predicted boundaries are in `[TILE_ID]/contours/v_[VERSION]/[TILE_ID].geojson`.

### Accuracy computation

For predicted vectorized field boundaries accuracy metrics are computed with the help of following steps.

1. Finding Ground Truth data to compare predicted to. 

For France there is https://geoservices.ign.fr/rpg where crop field boundary data is 
stored (vector format). 

Chose the region the tile is covering.

2. Download the cadastre data.

The command:
``` sh
  python src/other/download_cadastre.py
```
The results of the command is saved vector data for the tile region.

3. Convert vector data to raster and compute metrics (accuracy, Intersection over Union, ...).

The command:
``` sh
  python src/other/accuracy_computation.py
```
The output is vectorized field boundaries (Ground Truth - cadastre, Predicted - inference pipeline generated) and
`metrics_v[VERSION].csv` file with computed metrics for every patch (sub-tile) and mean of them.


## Implementation details

### Data preprocessing implementation

Preprocessing workflow diagram is available under `/visuals` :

![Preprocessing Workflow](visuals/data_preprocess_workflow.png)

### Inference implementation

Inference diagram is vailable under `/visuals` :
![Inference Workflow](visuals/inference_workflow.png)

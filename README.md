# Niva Project

Field delineation using the Resunet-a model.

## About The Project

This projects stems from the sentine-hub [field delineation project](https://github.com/sentinel-hub/field-delineation).

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

#### 4. Inference

TODO inference guide

## Implementation details

### Data preprocessing implementation

Preprocessing workflow diagram is available under `/visuals` :

![Preprocessing Workflow](visuals/data_preprocess_workflow.png)

### Training implementation

TODO add training diagram

### Inference implementation

TODO add inference diagram

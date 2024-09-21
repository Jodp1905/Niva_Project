<br />

<div align="center">
  <h1 align="center">Niva Project</h1>
  <p align="center">
    Field delineation using the Resunet-a model.
  </p>
</div>

# About The Project

This projects stems from the sentine-hub [field delineation project](https://github.com/sentinel-hub/field-delineation).

# Getting Started

This section will guide you on setting up the project and executing the full deep learning training and inference pipeline

## Prerequisites

### Python
This project uses python 3.10, that can be installed alongside other versions.
It is strongly advised to use a virtual environment.

* add python repository
  ```sh
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  ```

* install python 3.10
  ```sh
  sudo apt install python3.10 python3.10-venv python3.10-dev
  python3.10 --version
  ```

* create and activate virtual environment
  ```sh
  python3.10 -m venv /path/to/niva-venv
  source /path/to/niva-venv/bin/activate
  ```

### GEOS
The [geos library](https://github.com/libgeos/geos) is required by some python modules used.

* install with apt
  ```sh
  sudo apt update
  sudo apt-get install libgeos-dev
  ```
* to install from source, see https://github.com/libgeos/geos/blob/main/INSTALL.md

### The ai4boundaries dataset

This project uses the ai4boundaries dataset as the source for the training.
Learn more about ai4boundaries on [this repository](https://github.com/waldnerf/ai4boundaries)

## Installation

### 1. Clone the repository
```sh
git clone https://github.com/Jodp1905/Niva_Project.git
```

### 2. Set the environment variable NIVA_PROJECT_DATA_ROOT

All data used for training as well as the trained models will be saved at the path indicated by NIVA_PROJECT_DATA_ROOT.

```
niva_project_data   <---   The path to this folder 
│
└── sentinel2
    ├── images
    └── masks
```

Export it using :

```sh
export NIVA_PROJECT_DATA_ROOT=/path/to/ai4boundaries_dataset
```

You may also add the line to your ~/.bashrc file for convenience.

### 3. Install the python environment

You can set up the environment using the **requirements.txt** file at the root of the project.

* with your venv activated
  ```sh
  pip install -r requirements.txt
  ```

## Usage

This sections provides an overview on how you can get started and run the full field delineation training pipeline from end to end.
For more details about the implementation and the parameters you can tune, see the [detailed description](#detailed-description) section.

### Preprocessing pipeline

Data preprocessing can be executed using the `main_preprocessing` python script under /scripts :

```sh
python3 main_preprocessing.py
```

### Training

Once the datasets are created, you can run the model training using the `training.py` python script under /script with a training name of you choice :

```sh
python3 training.py <training-name>
```

Once the training has been executed, use the training name and the `main_analyze` script under /utils to generate loss/accuracy plots as well as a textual description of hyperparameters, memory usage, model size, and more !

```sh
python3 main_analyze.py
```

### Inference

TODO inference guide

# Implementation details

## Data preprocessing implementation

Preprocessing workflow diagram is available under /visuals :

![Preprocessing Workflow](visuals/data_preprocess_workflow.png)

## Training implementation

TODO add training diagram

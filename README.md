<br />

<div align="center">
  <h1 align="center">Niva Project</h1>
  <p align="center">
    I/O Analysis of a deep learning pipeline for field delineation using the resunet-A model.
  </p>
</div>

# About The Project

TODO

# Getting Started

This section will guide you on setting up the project and executing the full deep learning pipeline.  

## Prerequisites

### Python
This project uses python 3.10, that can be installed alongside other versions. It is strongly advised to use a virtual environment.

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

This project uses the ai4boundaries dataset as the source of the preprocessing pipeline.
You can download the it using instructions from [this repository](https://github.com/waldnerf/ai4boundaries)

## Installation

### 1. Clone the repository
```sh
git clone https://github.com/Jodp1905/Niva_Project.git
```

### 2. Set the environment variable NIVA_PROJECT_DATA_ROOT

It should be set to the root of the ai4boundaries dataset, which should look like this :

```
ai4boundaries_dataset   <---   The path to this folder 
├── orthophoto
│   ├── images
│   └── masks
├── samples
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

This sections provides an overview on how you can quickly run the full field delineation training pipeline from end to end.

For more details about the implementation and the many parameters you can tune, see the [detailed description](#detailed-description) section.

### Preprocessing pipeline

Data preprocessing can be executed using the `main_preprocessing` python script under /scripts :

```sh
python3 main_preprocessing.py
```

[This image](./data_preprocess_workflow.png) describes in detail the process of creating the tensorflow datasets that can be used for training from the initial ai4boundaries dataset.

### Training

Once the datasets are created, you can run the model training using the `training.py` python script under /script with a training name of you choice :

```sh
python3 training.py <training-name>
```

### Post processing

Once the training has been executed, use the training name and the `main_analyze` script under /utils to generate loss/accuracy plots as well as a textual description of hyperparameters, memory usage, model size, and more !

```sh
python3 main_analyze.py
```

# Implementation details

TBD

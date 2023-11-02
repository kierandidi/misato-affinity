<div align="center">

# MISATO Affinity Predictions

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.8+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

</div>
 
## :purple_heart: Community

Want to get hands-on for drug discovery using AI?

[Join our discord server!](https://discord.gg/tGaut92VYB)

## :rocket:  About

In this repository we provide the code for the binding affinity prediction task described in [our paper](https://www.biorxiv.org/content/10.1101/2023.05.24.542082v2). For the main dataset and instructions how to download it visit the [main repository site](https://github.com/t7morgen/misato-dataset)

## :computer:  Environment setup

### 1. Conda environment
Create a conda environment for the project via
```bash
make venv # will create a cpu environment
# NOTE: This will simply call
#  conda env create --prefix=./venv -f requirements/env.yml

# For a gpu environment call
#  make name=venv_gpu sys=gpu venv
#  conda activate ./venv_gpu

# For a Mac m1 environment call
#  make name=venv sys=m1 venv
#  conda activate ./venv

# To activate the environment, use:
conda activate ./venv
```

After this, install the local dependencies via `pip install -e .`, executing this command from the project directory.

### 2. Environment variables

Set environment variables for you system in a `.env` file at the project directory
(same as this README.) Specify the following variables:

```
# Path to the general data directory
DATA_PATH=data/

# Path to the directory where run outputs will be stored
RUNS_PATH=<path_to_your_runs_directory>

# Path to the root of the project directory
PROJECT_PATH=<path_to_your_project_directory>

# Path to the .hdf5 file containing Molecular Dynamics (MD) data
MD_PATH=<path_to_your_md_data_file>

# Path to the .hdf5 file containing (QM) data
QM_PATH=<path_to_your_qm_data_file>

# Path to the .hdf5 file containing the h5 of the preprocessed invariant graphs
INVARIANT_GRAPH_PATH=<path_to_your_graph_data_file>

# Path to the .h5 file containing affinity data
AFFINITY_PATH=data/affinity_data.h5

# Path to the .pickle file containing all possible available protein-ligand pairs
PAIR_PATH_TRAIN=data/train_pairs.pickle
PAIR_PATH_TEST=data/test_pairs.pickle
PAIR_PATH_VAL=data/val_pairs.pickle

# Your Weights & Biases API key
WANB_API_KEY="<your_wandb_api_key>"

# Your Weights & Biases entity (username or team name)
WANDB_ENTITY="<your_wandb_entity>"

# Your Weights & Biases project name
WANDB_PROJECT="<your_wandb_project_name>"
```

### 3. Config variables

This project uses Hydra for configuration management. Adjust the parameters in the `configs`
directory to your setup to run this project or adjust them via the command line (see below).

## :file_folder: MISATO files and Preprocessing

The MISATO h5 files can be downloaded like this:

```bash
wget -O data/MD.hdf5 https://zenodo.org/record/7711953/files/MD.hdf5
wget -O data/QM.hdf5 https://zenodo.org/record/7711953/files/QM.hdf5
```
You can download a preprocessed h5 file containing the MD adaptability and reference coordinates from here:
https://syncandshare.lrz.de/getlink/fiVDroRT3k1cy1krpNp9Mj/adaptability_MD.hdf5

The preprocessed graphs for the dataloader can also be downloaded.

Invariant graph:
https://syncandshare.lrz.de/getlink/fiGQ67kokEWG28rJ3fzCGt/preprocessed_graph_invariant.h5

Alternatively, generate a h5 file containing the adaptability values from the MD.hdf5 file by running the preprocessing:
```
python src/start_preprocessing.py
```
The preprocessing scripts for the graphs can be found in src/data/processing/.


## :chart_with_upwards_trend: Experiment logging with wandb

To log to wandb, you will first need to log in. To do so, simply install wandb via pip
with `pip install wandb` and call `wandb login` from the commandline.

If you are already logged in and need to relogin for some reason, use `wandb login --relogin`.

## :mechanical_arm:	Training a model with pytorch lightning and logging on wandb

To run a model simply use

```
python src/train.py name=<YOUR_RUN_NAME>
```
By default, `train.py` uses Weights&Biases logging via the credentials you provided in your `.env` file. If you do not pass a name for the run, the default name `test` will be used and logging will be disabled. If you give your run a name different than `test`, WandB logging will be enabled and the run will be logged to your WandB account with the name you gave it.

For the invariant case, you could run a training run like:
```
python src/train.py name=<YOUR_RUN_NAME> model=gcn datamodule=md_datamodule_invariant
```

For the equivariant case, you could run a training run like (use either `egnn` or `gvp`):
```
python src/train.py name=<YOUR_RUN_NAME> model=egnn datamodule=md_datamodule
```


To use parameters that are different from the default parameters in `src/configs/config.yaml`
you can simply provide them in the command line call. For example:

```
python src/train.py name=<YOUR_RUN_NAME> trainer.epochs=100
```

By default, we run a test run after training. If you want to disable that, you need to pass the config `model_test=False` when starting the training.


To configure extra things such as logging, use
```
# LOGGING
# For running at DEBUG logging level:
#  (c.f. https://hydra.cc/docs/tutorials/basic/running_your_app/logging/ )
## Activating debug log level only for loggers named `__main__` and `hydra`
python src/train.py 'hydra.verbose=[__main__, hydra]'
## Activating debug log level for all loggers
python src/train.py hydra.verbose=true

# PRINTING CONFIG ONLY
## Print only the job config, then return without running
python src/train.py --cfg job

# GET HELP
python src/train.py --help
```

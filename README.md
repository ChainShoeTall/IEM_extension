# Paper name

This repository contains all the configurations and code used in the experiments (except traditional image enhancement methods such as GC for now).

We make significant changes and additions to rPPG-Toolbox by ubicomplab. The major changes include

1. more flexible training setting by make use of/ extend the options of yaml, such as 

    - Multiprocessing: control number of dataloader for each dataset, and control number of processes in preprocessing the dataset

    - Training: select loss function, more flexible range selection

2. materialization: produce a number of output files are each training/testing, where the parent directory is controlled by yaml and the output directory itself is controlled by model name (in yaml) and system time. Output files include:

    - a copy of settings yaml file
    - final models (in 'train-and-test' mode, and for Physnet-Raw and Physnet_Raw_enhSCI only)
    - console log as txt file
    - HR predictions in CSV and a simple plot in PNG (for each inference method)

3. upgraded the face detection method to a mediapipe implementation

4. additional trainer files and folders (such as enhancement/SCI) which are related to our experiments

5. config files and produced models in our experiments

6. refactoring

# Setup

Please follow the original instructions of rppg-toolbox (`README_original.md`) to setup an environment with all neede packages installed. (`The requirements.txt` is updated to include mediapipe)

for mediapipe, may need to run the following install to solve problems related to import cv2 
    
    pip install "opencv-python-headless<4.3"

Some upgrades to programming environment is possible, our code can be run in the following system configuration:

GPU: GeForce RTX3080 (Driver version 522.06)

OS: Ubuntu 22.04 LTS

Software: python 3.8.16 + pytorch 2.0.1 + cuda 11.8

# Module

The framework contains mainly 4 modules: 

1. Deep-learning based rPPG extraction model (`neural_method/model`): This contains the structure and framework of the rPPG extraction model.
Tts weights is stored in (`production/models/ubnfc_physnet_NPLoss.pth`); 

2. Image Enhancement model (`enhancement/model`): This contains the structure and framework of the Image enhancement model (Self Calibration Image (SCI) Enhancement model). It's weights (`production/models/SCI_bhlow_another_teston_*.pth`);

3. Model trainer (`neural_methods/Trainer/PhysnetTrainer_*.py`): This contains the Initialization/training/validation/testing process of the framework;

4. Data loader (`dataset/data_loader/*Loader.py`): This contains the torch dataloader for different dataset. It consists of data preprocessing (including face cropping, resizing and normalization), dataset splitting, etc.

# Main.py

`main.py` contains the major processes of the framework.

1. The argument is parsed to load the configurations in `yaml` files. The `yaml` file is parsed by `config.py`, which also contains the default values for each parameter. The configuration contains all the information needed for running, including:
    - The Dataset used for training/validation/testing, and its descriptions like FPS, Path, Splitting info, Preprocessing info (face cropping parameters)

    - Details for training, e.g. learning rate, loss function, batch size, epochs, etc.


2. Load the dataset. If `toolbox_mode` is `only_test` or `unsupervised_method`, it will only load the dataset for testing; Otherwise if `toolbox_mode` is `train_and_test`, it will load the dataset  for training, validation and testing.

3. After data loading, it will assign the trainer for training/validation/testing. The details of the training/validation/testing is stored in Model Trainer.

# YAML file

YAML file contains the configurations for running. Here we only introduce some important and 'misleading' parameters, for details please refer to the paper of rPPG-toolbox.

+ TOOLBOX_MODE (!): `train_and_test` for training and testing; `only_test` for test only; `unsupervised_method` for traditional rPPG extraction methods.

`TRAIN`, `VALID` and `TEST` contains similar contents, which include:

+ LOSS: Loss function, Negative Pearson (NP) or Time-shifted NP (SNP).

+ MODEL_FILE_NAME: The file name to save the trained model weigths (.pth files).

+ MODEL_PATH: Path of the pre-trained physnet model weights, this is only needed to re-training of the physnet model. 

+ ENHANCEMODEL_PATH: Path of the image enhancement model weights (default to be `enhancement/medium.pt`). This is only needed for IEM + PhysNet model training.

+ DATA: the details of the dataset.

    + FS: Frame per second (FPS) of the video recording.

    + DATASET: Dataset name.

    + SPECIFIC: (Only for BH dataset) Specifies the illumination condition used (`low/middle/high` for each illumination levels, `default` for all levels).

    + BEGIN, END, EXCEPT: These 3 parameters together determines the portion of the dataset used. 0.0 <= BEGIN <= END <= 1.0, and EXCEPT is set to False to include the portion between BEGIN and END, and True to exclude the portion between BEGIN and END.

    + PREPROCESS: the details for preprocessing including normalization and face cropping. As we employ the 'Difference Normalization' in the trainer, the DATA_TYPE is always set to ['Raw']. 

+ USE_LAST_EPOCH: True for using the last epoch, and False to use the epoch with best performance in validation. If TEST.USE_LAST_EPOCH is true, we only need to leave VALID.DATA.DATASET to make them work.

+ MODEL: Details of the model

    + NAME (Important!): Determines which model training to use.

    + FRAME_NUM: The length of each frame clip, needed to PhysNet.

+ INFERENCE: The details for `only_test` toolbox mode.

    + MODEL_PATH: The weights for rPPG extraction model, load for testing.

    + ENHANCEMODEL_PATH: The weights for image enhancement model, load for testing only for IEM + PhysNet testing.


# Running

The only entry point of the toolbox is main.py, so we can use this format in each run

    python main.py --config_file <yaml file>

Training yamls and produced model are provided in `production/` directory. 

- You need set which `model` to be tested, which is stored in `MODEL.NAME`. 
    - `PhysnetRaw` for PhysNet only.
    - `Physnet-SCI` for IEM + PhysNet.

- The `test*` and `unsupervised*` configs can be used without any training. 

- For testing, you need to set the `toolbox_mode` to `only_test`, and config the model path to be tested, which is stored in `INFERENCE.MODEL_PATH`; For IEM + PhysNet, you also need config the IEM model path, which is stored in `INFERENCE.ENHANCEMODEL_PATH`.

To reproduce the training, first run `ubfc_physnet_settings.yamls` to produce the Baseline Physnet, then run `production/configs/bhlow_part1_SCI_finetune_settings.yaml` and `production/configs/bhlow_part2_SCI_finetune_settings.yaml`. For the finetuning of SCI model, may replace the underlying freezed Physnet model.

Note that we have performed a careful search of hyperparameters, and the SCI model is highly depenedent of the underlying Physnet. Hence another grid search is needed if the Physnet model is replaced.

# Further modificatons

We suggest writing new dataloader files and new trainer files for experiments but not modifying given files. 

For materialization code it is scattered in several files

- `main.py` (for copying yml and produce log file)

- each trainer file (for storing final models)

- `evaluation/metrics.py` (for HR prediction materialization of nerual methods)

- `unsupervised_methods/unsupervised_predictor.py` (for HR prediction materialization of unsupervised methods)

For additional configs, need to update `config.py` and any related files following the flow of the config throughout the main program.
'''config file to control the training and testing
NOT ALL configs are used in actual program, read the trainer files/ unsupervised predictor files for details
'''
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Shutao & Tony
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------\
_C.TOOLBOX_MODE = "" # "only_test" for testing only; "train_and_test" for model training and testing
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.LOSS = "NP" # Loss function(str):"NP" for Negative Pearson Correlation; "SNP" for Time-shifted Negative Pearson Correlation.
_C.TRAIN.LR = 1e-4
_C.TRAIN.MODEL_FILE_NAME = '' # Saved trained model name (.pth)

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.EPS = 1e-4
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Train.Data settings
_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.FS = 0
_C.TRAIN.DATA.DATA_PATH = ''
_C.TRAIN.DATA.EXP_DATA_NAME = ''
_C.TRAIN.DATA.CACHED_PATH = 'PreprocessedData'
_C.TRAIN.DATA.FILE_LIST_PATH = os.path.join(_C.TRAIN.DATA.CACHED_PATH, 'DataFileLists')
_C.TRAIN.DATA.DATASET = ''
_C.TRAIN.DATA.DO_PREPROCESS = False
_C.TRAIN.DATA.DATA_FORMAT = 'NDCHW'
_C.TRAIN.DATA.BEGIN = 0.0
_C.TRAIN.DATA.END = 1.0
_C.TRAIN.DATA.SPECIFIC = 'default'
_C.TRAIN.DATA.EXCEPT = False



# Train Data preprocessing
_C.TRAIN.DATA.PREPROCESS = CN()
_C.TRAIN.DATA.PREPROCESS.DO_CHUNK = True
_C.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH = 128
_C.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION = False # Default as false (for Mediapipe, it can treat image sequence as stream)
_C.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 128
_C.TRAIN.DATA.PREPROCESS.CROP_FACE = True
_C.TRAIN.DATA.PREPROCESS.LARGE_FACE_BOX = False
_C.TRAIN.DATA.PREPROCESS.LARGE_BOX_COEF = 1.0
_C.TRAIN.DATA.PREPROCESS.W = 72
_C.TRAIN.DATA.PREPROCESS.H = 72
_C.TRAIN.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TRAIN.DATA.PREPROCESS.MULTI_PROCESS_QUOTA = 4

# Train model settings
_C.TRAIN.MODEL_PATH = ''
_C.TRAIN.ENHANCEMODEL_PATH = ''
_C.TRAIN.PREMODEL_PATH = ''
_C.ENHANCEMODEL_NAME = 'SCI'



# -----------------------------------------------------------------------------
# Valid settings
# -----------------------------------------------------------------------------\
_C.VALID = CN()
# Valid.Data settings
_C.VALID.DATA = CN()
_C.VALID.DATA.FS = 30
_C.VALID.DATA.DATA_PATH = ''
_C.VALID.DATA.EXP_DATA_NAME = ''
_C.VALID.DATA.CACHED_PATH = 'PreprocessedData'
_C.VALID.DATA.FILE_LIST_PATH = os.path.join(_C.VALID.DATA.CACHED_PATH, 'DataFileLists')
_C.VALID.DATA.DATASET = ''
_C.VALID.DATA.DO_PREPROCESS = False
_C.VALID.DATA.DATA_FORMAT = 'NDCHW'
_C.VALID.DATA.BEGIN = 0.0
_C.VALID.DATA.END = 1.0
_C.VALID.DATA.SPECIFIC = 'default'
_C.VALID.DATA.EXCEPT = False


# Valid Data preprocessing
_C.VALID.DATA.PREPROCESS = CN()
_C.VALID.DATA.PREPROCESS.DO_CHUNK = True
_C.VALID.DATA.PREPROCESS.CHUNK_LENGTH = 128
_C.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 128
_C.VALID.DATA.PREPROCESS.CROP_FACE = True
_C.VALID.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.VALID.DATA.PREPROCESS.LARGE_BOX_COEF = 1.0
_C.VALID.DATA.PREPROCESS.W = 128
_C.VALID.DATA.PREPROCESS.H = 128
_C.VALID.DATA.PREPROCESS.DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.LABEL_TYPE = ''
_C.VALID.DATA.PREPROCESS.MULTI_PROCESS_QUOTA = 4



# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------\
_C.TEST = CN()
_C.TEST.METRICS = []
_C.TEST.USE_LAST_EPOCH = True
# Test.Data settings
_C.TEST.DATA = CN()
_C.TEST.DATA.FS = 0
_C.TEST.DATA.DATA_PATH = ''
_C.TEST.DATA.EXP_DATA_NAME = ''
_C.TEST.DATA.CACHED_PATH = 'PreprocessedData'
_C.TEST.DATA.FILE_LIST_PATH = os.path.join(_C.TEST.DATA.CACHED_PATH, 'DataFileLists')
_C.TEST.DATA.DATASET = ''
_C.TEST.DATA.DO_PREPROCESS = False
_C.TEST.DATA.DATA_FORMAT = 'NDCHW'
_C.TEST.DATA.BEGIN = 0.0
_C.TEST.DATA.END = 1.0
_C.TEST.DATA.SPECIFIC = 'default'
_C.TEST.DATA.EXCEPT = False



# Test Data preprocessing
_C.TEST.DATA.PREPROCESS = CN()
_C.TEST.DATA.PREPROCESS.DO_CHUNK = True
_C.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 128
_C.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 128
_C.TEST.DATA.PREPROCESS.CROP_FACE = True
_C.TEST.DATA.PREPROCESS.LARGE_FACE_BOX = False
_C.TEST.DATA.PREPROCESS.LARGE_BOX_COEF = 1.0
_C.TEST.DATA.PREPROCESS.W = 72
_C.TEST.DATA.PREPROCESS.H = 72
_C.TEST.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TEST.DATA.PREPROCESS.MULTI_PROCESS_QUOTA = 4
_C.TEST.SAVE_RESULT_NAME = None


# -----------------------------------------------------------------------------
# Unsupervised method settings
# -----------------------------------------------------------------------------\
_C.UNSUPERVISED = CN()
_C.UNSUPERVISED.METHOD = []
_C.UNSUPERVISED.METRICS = []
# Unsupervised.Data settings
_C.UNSUPERVISED.DATA = CN()
_C.UNSUPERVISED.DATA.FS = 0
_C.UNSUPERVISED.DATA.DATA_PATH = ''
_C.UNSUPERVISED.DATA.EXP_DATA_NAME = ''
_C.UNSUPERVISED.DATA.CACHED_PATH = 'PreprocessedData'
_C.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(_C.UNSUPERVISED.DATA.CACHED_PATH, 'DataFileLists')
_C.UNSUPERVISED.DATA.DATASET = ''
_C.UNSUPERVISED.DATA.DO_PREPROCESS = False
_C.UNSUPERVISED.DATA.DATA_FORMAT = 'NDCHW'
_C.UNSUPERVISED.DATA.BEGIN = 0.0
_C.UNSUPERVISED.DATA.END = 1.0
_C.UNSUPERVISED.DATA.SPECIFIC = 'default'
_C.UNSUPERVISED.DATA.EXCEPT = False
_C.UNSUPERVISED.DATA.PABP_DEVICE = 'ipad'
_C.UNSUPERVISED.DATA.PABP_ILLUMINATION = 100
_C.UNSUPERVISED.DATA.PABP_DISTANCE = 0
_C.UNSUPERVISED.DATA.PABP_ET = 4
_C.UNSUPERVISED.DATA.PABP_GAIN = 5


# Unsupervised Data preprocessing
_C.UNSUPERVISED.DATA.PREPROCESS = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.DO_CHUNK = True
_C.UNSUPERVISED.DATA.PREPROCESS.CHUNK_LENGTH = 128
_C.UNSUPERVISED.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.UNSUPERVISED.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 128
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE = True
_C.UNSUPERVISED.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.UNSUPERVISED.DATA.PREPROCESS.LARGE_BOX_COEF = 1.0
_C.UNSUPERVISED.DATA.PREPROCESS.W = 72
_C.UNSUPERVISED.DATA.PREPROCESS.H = 72
_C.UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE = ['']
_C.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE = ''
_C.UNSUPERVISED.DATA.PREPROCESS.MULTI_PROCESS_QUOTA = 4

# For PABP dataset only
_C.TRAIN.DATA.PABP_DEVICE = 'ipad'
_C.TRAIN.DATA.PABP_ILLUMINATION = 100
_C.TRAIN.DATA.PABP_DISTANCE = 0

_C.VALID.DATA.PABP_DEVICE = 'ipad'
_C.VALID.DATA.PABP_ILLUMINATION = 100
_C.VALID.DATA.PABP_DISTANCE = 0

_C.TEST.DATA.PABP_DEVICE = 'ipad'
_C.TEST.DATA.PABP_ILLUMINATION = 100
_C.TEST.DATA.PABP_DISTANCE = 0
_C.TEST.DATA.PABP_ET = 4
_C.TEST.DATA.PABP_GAIN = 5

### -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.MODEL_DIR = 'PreTrainedModels'

# Specific parameters for physnet parameters
_C.MODEL.PHYSNET = CN()
_C.MODEL.PHYSNET.FRAME_NUM = 128

# -----------------------------------------------------------------------------
# Model Settings for TS-CAN
# -----------------------------------------------------------------------------
_C.MODEL.TSCAN = CN()
_C.MODEL.TSCAN.FRAME_DEPTH = 10

# -----------------------------------------------------------------------------
# Model Settings for EfficientPhys
# -----------------------------------------------------------------------------
_C.MODEL.EFFICIENTPHYS = CN()
_C.MODEL.EFFICIENTPHYS.FRAME_DEPTH = 10

# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.EVALUATION_METHOD = 'FFT'
_C.INFERENCE.MODEL_PATH = ''
_C.INFERENCE.ENHANCEMODEL_PATH = ''
_C.INFERENCE.TEST_FOLD = None
# -----------------------------------------------------------------------------
# Device settings
# -----------------------------------------------------------------------------
_C.DEVICE = "cuda:0"
_C.NUM_OF_GPU_TRAIN = 1

# -----------------------------------------------------------------------------
# Log settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.PATH = "runs/exp"
# -----------------------------------------------------------------------------
# Additional settings
# -----------------------------------------------------------------------------
_C.DATALOADER_WORKERS = 4

# -----------------------------------------------------------------------------
# Traditional Image Enhancement Settings
# -----------------------------------------------------------------------------
_C.TRAD = CN()
_C.TRAD.NAME = ""
_C.TRAD.GAMMA_VALUE = 2.5
# -----------------------------------------------------------------------------

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> Merging a config file from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def produce_filelist_name(slist: list):
    '''return the formatted data file path
    slist should be in order
    [ DATASET,
    PREPROCESS.W,
    PREPROCESS.H,
    PREPROCESS.CHUNK_LENGTH,
    PREPROCESS.DATA_TYPE,
    PREPROCESS.LABEL_TYPE,
    PREPROCESS.LARGE_FACE_BOX,
    PREPROCESS.LARGE_BOX_COEF,
    PREPROCESS.DYNAMIC_DETECTION,
    PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
    ]
    '''
    res = "_".join([
        slist[0],
        f"SizeW{slist[1]}",
        f"SizeH{slist[2]}",
        f"ClipLength{slist[3]}",
        f"DataType{'_'.join(slist[4])}",
        f"LabelType{slist[5]}",
        f"Large_box{slist[6]}",
        f"Large_size{slist[7]}",
        f"Dyamic_Det{slist[8]}",
        f"det_len{slist[9]}",
        # slist[10]
    ])
    # not here! 
    if (len(slist) == 11) and (slist[10] != 0):
        res = "_".join([res, str(slist[10])]) # for distance of PABP
    if (len(slist) == 13) and (slist[10] == 0):
        res = "_".join([res, "ET" + str(slist[11]), "GAIN" + str(slist[12])])
    return res

def produce_data_filelist_csv(file_list_path, exp_data_name, specific, alist:list):
    '''return the filename string with .csv to save filelist
    alist: [begin, end, except]
    '''
    begin, end, isexcept = alist
    return os.path.join(file_list_path, exp_data_name  + f"_{specific}" + '_' + \
            str(begin) + '_' + \
            str(end) + \
            ("_except" if isexcept else "") + 
            '.csv')

def update_config(config, args):

    # store default file list path for checking against later
    default_TRAIN_FILE_LIST_PATH = config.TRAIN.DATA.FILE_LIST_PATH
    default_VALID_FILE_LIST_PATH = config.VALID.DATA.FILE_LIST_PATH
    default_TEST_FILE_LIST_PATH = config.TEST.DATA.FILE_LIST_PATH
    default_UNSUPERVISED_FILE_LIST_PATH = config.UNSUPERVISED.DATA.FILE_LIST_PATH

    # update flag from config file
    _update_config_from_file(config, args.config_file)
    config.defrost()

    # first of all my own updates
    from datetime import datetime
    current_time = datetime.now().strftime(r"%Y%b%d_%H:%M:%S")
    config.LOG.PATH = os.path.join(config.LOG.PATH, "_".join([config.MODEL.NAME, config.TEST.DATA.DATASET,current_time]))
    
    # UPDATE TRAIN PATHS
    if config.TRAIN.DATA.FILE_LIST_PATH == default_TRAIN_FILE_LIST_PATH:
        config.TRAIN.DATA.FILE_LIST_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, 'DataFileLists')

    if config.TRAIN.DATA.EXP_DATA_NAME == '':
        newname = produce_filelist_name([
            config.TRAIN.DATA.DATASET,
            config.TRAIN.DATA.PREPROCESS.W,
            config.TRAIN.DATA.PREPROCESS.H,
            config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,
            config.TRAIN.DATA.PREPROCESS.DATA_TYPE,
            config.TRAIN.DATA.PREPROCESS.LABEL_TYPE,
            config.TRAIN.DATA.PREPROCESS.LARGE_FACE_BOX,
            config.TRAIN.DATA.PREPROCESS.LARGE_BOX_COEF,
            config.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION,
            config.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY,
            config.TRAIN.DATA.PABP_DISTANCE
        ])
        config.TRAIN.DATA.EXP_DATA_NAME = newname
    config.TRAIN.DATA.CACHED_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, config.TRAIN.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.TRAIN.DATA.FILE_LIST_PATH) # FILE_LIST_PATH = .../DataFileLists => ext is None
    if not ext: # no file extension
        config.TRAIN.DATA.FILE_LIST_PATH = \
            produce_data_filelist_csv(config.TRAIN.DATA.FILE_LIST_PATH, config.TRAIN.DATA.EXP_DATA_NAME, config.TRAIN.DATA.SPECIFIC,
                          [str(config.TRAIN.DATA.BEGIN), str(config.TRAIN.DATA.END), config.TRAIN.DATA.EXCEPT])
    elif ext != '.csv':
        raise ValueError('TRAIN dataset FILE_LIST_PATH must either be a directory path or a .csv file name')
    
    if ext == '.csv' and config.TRAIN.DATA.DO_PREPROCESS:
        raise ValueError('User specified TRAIN dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing TRAIN dataset FILE_LIST_PATH .csv file.')

    # ===============================Setting for VALID====================================
    # If only test, only need to assign VALID.DATA.DATASET and will skip VLID
    if not config.TEST.USE_LAST_EPOCH and config.VALID.DATA.DATASET is not None:
        # UPDATE VALID PATHS
        if config.VALID.DATA.FILE_LIST_PATH == default_VALID_FILE_LIST_PATH:
            config.VALID.DATA.FILE_LIST_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, 'DataFileLists')

        if config.VALID.DATA.EXP_DATA_NAME == '':
            config.VALID.DATA.EXP_DATA_NAME = produce_filelist_name([
                config.VALID.DATA.DATASET,
                config.VALID.DATA.PREPROCESS.W,
                config.VALID.DATA.PREPROCESS.H,
                config.VALID.DATA.PREPROCESS.CHUNK_LENGTH,
                config.VALID.DATA.PREPROCESS.DATA_TYPE,
                config.VALID.DATA.PREPROCESS.LABEL_TYPE,
                config.VALID.DATA.PREPROCESS.LARGE_FACE_BOX,
                config.VALID.DATA.PREPROCESS.LARGE_BOX_COEF,
                config.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION,
                config.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
            ])
        config.VALID.DATA.CACHED_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, config.VALID.DATA.EXP_DATA_NAME)

        name, ext = os.path.splitext(config.VALID.DATA.FILE_LIST_PATH)
        if not ext:  # no file extension
            config.VALID.DATA.FILE_LIST_PATH = produce_data_filelist_csv(config.VALID.DATA.FILE_LIST_PATH, config.VALID.DATA.EXP_DATA_NAME, config.VALID.DATA.SPECIFIC,
                          [str(config.VALID.DATA.BEGIN), str(config.VALID.DATA.END), config.VALID.DATA.EXCEPT])
        elif ext != '.csv':
            raise ValueError('VALIDATION dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

        if ext == '.csv' and config.VALID.DATA.DO_PREPROCESS:
            raise ValueError('User specified VALIDATION dataset FILE_LIST_PATH .csv file already exists. \
                            Please turn DO_PREPROCESS to False or delete existing VALIDATION dataset FILE_LIST_PATH .csv file.')
    elif not config.TEST.USE_LAST_EPOCH and config.VALID.DATA.DATASET is None:
        raise ValueError('VALIDATION dataset is not provided despite USE_LAST_EPOCH being False!')

    # ===============================Setting for TEST====================================
    if config.TEST.DATA.FILE_LIST_PATH == default_TEST_FILE_LIST_PATH:
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, 'DataFileLists')

    if config.TEST.DATA.EXP_DATA_NAME == '':
        if config.TEST.DATA.DATASET == "PABP_Raw":
            config.TEST.DATA.EXP_DATA_NAME = produce_filelist_name([
                config.TEST.DATA.DATASET,
                config.TEST.DATA.PREPROCESS.H,
                config.TEST.DATA.PREPROCESS.W,
                config.TEST.DATA.PREPROCESS.CHUNK_LENGTH,
                config.TEST.DATA.PREPROCESS.DATA_TYPE,
                config.TEST.DATA.PREPROCESS.LABEL_TYPE,
                config.TEST.DATA.PREPROCESS.LARGE_FACE_BOX,
                config.TEST.DATA.PREPROCESS.LARGE_BOX_COEF,
                config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION,
                config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY,
                config.TEST.DATA.PABP_DISTANCE,
                config.TEST.DATA.PABP_ET,
                config.TEST.DATA.PABP_GAIN
            ])
        else:
            config.TEST.DATA.EXP_DATA_NAME = produce_filelist_name([
                config.TEST.DATA.DATASET,
                config.TEST.DATA.PREPROCESS.H,
                config.TEST.DATA.PREPROCESS.W,
                config.TEST.DATA.PREPROCESS.CHUNK_LENGTH,
                config.TEST.DATA.PREPROCESS.DATA_TYPE,
                config.TEST.DATA.PREPROCESS.LABEL_TYPE,
                config.TEST.DATA.PREPROCESS.LARGE_FACE_BOX,
                config.TEST.DATA.PREPROCESS.LARGE_BOX_COEF,
                config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION,
                config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY,
                config.TEST.DATA.PABP_DISTANCE,
            ])
    config.TEST.DATA.CACHED_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, config.TEST.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.TEST.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        config.TEST.DATA.FILE_LIST_PATH = produce_data_filelist_csv(config.TEST.DATA.FILE_LIST_PATH, config.TEST.DATA.EXP_DATA_NAME, config.TEST.DATA.SPECIFIC,
                          [str(config.TEST.DATA.BEGIN), str(config.TEST.DATA.END), config.TEST.DATA.EXCEPT])
    elif ext != '.csv':
        raise ValueError('TEST dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.TEST.DATA.DO_PREPROCESS:
        raise ValueError('User specified TEST dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing TEST dataset FILE_LIST_PATH .csv file.')
    
    # ===============================Setting for TEST====================================
    # UPDATE UNSUPERVISED PATHS
    if config.UNSUPERVISED.DATA.FILE_LIST_PATH == default_UNSUPERVISED_FILE_LIST_PATH:
        config.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(config.UNSUPERVISED.DATA.CACHED_PATH, 'DataFileLists')

    if config.UNSUPERVISED.DATA.EXP_DATA_NAME == '':
        config.UNSUPERVISED.DATA.EXP_DATA_NAME = produce_filelist_name([
            config.UNSUPERVISED.DATA.DATASET,
            config.UNSUPERVISED.DATA.PREPROCESS.H,
            config.UNSUPERVISED.DATA.PREPROCESS.W,
            config.UNSUPERVISED.DATA.PREPROCESS.CHUNK_LENGTH,
            config.UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE,
            config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE,
            config.UNSUPERVISED.DATA.PREPROCESS.LARGE_FACE_BOX,
            config.UNSUPERVISED.DATA.PREPROCESS.LARGE_BOX_COEF,
            config.UNSUPERVISED.DATA.PREPROCESS.DYNAMIC_DETECTION,
            config.UNSUPERVISED.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY,
            config.UNSUPERVISED.DATA.PABP_DISTANCE
        ])
    config.UNSUPERVISED.DATA.CACHED_PATH = os.path.join(config.UNSUPERVISED.DATA.CACHED_PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.UNSUPERVISED.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        config.UNSUPERVISED.DATA.FILE_LIST_PATH = produce_data_filelist_csv(config.UNSUPERVISED.DATA.FILE_LIST_PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME, config.UNSUPERVISED.DATA.SPECIFIC,
                          [str(config.UNSUPERVISED.DATA.BEGIN), str(config.UNSUPERVISED.DATA.END), config.UNSUPERVISED.DATA.EXCEPT])
    elif ext != '.csv':
        raise ValueError('UNSUPERVISED dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.UNSUPERVISED.DATA.DO_PREPROCESS:
        raise ValueError('User specified UNSUPERVISED dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing UNSUPERVISED dataset FILE_LIST_PATH .csv file.')


    config.MODEL.MODEL_DIR = os.path.join(config.MODEL.MODEL_DIR, config.TRAIN.DATA.EXP_DATA_NAME)
    config.freeze()
    return



def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config



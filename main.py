""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

# override global print to output into log if assigned
import builtins
output_file = None
_print = print
def print(*args, **kwargs):
    if output_file == None: # fall back to original print
        _print(*args, **kwargs)
    else:
        _print(*args, **kwargs)
        _print(*args, **kwargs, file=output_file, flush=True)
builtins.print = print

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser, please look at configs/ directory for examples"""
    parser.add_argument('--config_file', required=False,
                        default="configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    return parser

# ======own additions==================================
def assigntrainer(trstr):
    '''return the loader for the dataset specified by trstr
    current trstr choices: Physnet/Tscan/EfficientPhys/DeepPhys/Physnet-E
    for 4GB gpu: only Physnet
    testing: 
        Physnet-E -> IEM + PhysNet
        Physnet-R -> Retrain PhysNet
        Physnet-Rb2u -> Retrain PhysNet[BH] with UBFC
        Physnet-Ru2b -> Retrain PhysNet[UBFC] with BH
        Physnet-T -> Traditional IE + PhysNet
    '''
    if trstr == "Physnet":
        this_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif trstr == "Tscan":
        this_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif trstr == "EfficientPhys":
        this_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif trstr == "DeepPhys":
        this_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif trstr == "DeepPhys-Raw":
        this_trainer = trainer.DeepPhysTrainer_Raw.DeepPhysTrainer(config, data_loader_dict)
    elif trstr == "DeepPhys-SCI":
        this_trainer = trainer.DeepPhysTrainer_SCI.DeepPhysTrainer(config, data_loader_dict)
    elif trstr == "Physnet-E":
        this_trainer = trainer.PhysnetTrainer_Enhanced.PhysnetTrainer(config, data_loader_dict)
    elif trstr == "Physnet-SCI":
        this_trainer = trainer.PhysnetTrainer_SCI.PhysnetTrainer(config, data_loader_dict)
    elif trstr == "Physnet-Raw":
        this_trainer = trainer.PhysnetTrainer_Raw.PhysnetTrainerRaw(config, data_loader_dict)
    elif trstr  =="Physnet-Rb2u":
        this_trainer = trainer.PhysnetTrainer_Raw_b2u.PhysnetTrainer(config, data_loader_dict)
    elif trstr  =="Physnet-Ru2b":
        this_trainer = trainer.PhysnetTrainer_Raw_u2b.PhysnetTrainer(config, data_loader_dict)
    elif trstr == "Physnet-T":
        this_trainer = trainer.PhysnetTrainer_TradEnhance.PhysnetTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your trainer is Not Supported  Yet!')
    return this_trainer

def assignloader(dsstr):
    '''return the loader for the dataset specified by dsstr
    current dsstr choices: UBFC/PURE/SCAMPS/BH
    '''
    if dsstr == "COHFACE":
        raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, and SCAMPS.")
    elif dsstr == "UBFC":
        this_loader = data_loader.UBFCLoader.UBFCLoader
    elif dsstr == "PURE":
        this_loader = data_loader.PURELoader.PURELoader
    elif dsstr == "SCAMPS":
        this_loader = data_loader.SCAMPSLoader.SCAMPSLoader
    elif dsstr == "BH":
        this_loader = data_loader.BHLoader_multi.BHLoader
    elif dsstr == "PABP":
        this_loader = data_loader.PABPLoader.PABPLoader
    elif dsstr == "PABP_Raw":
        this_loader = data_loader.PABPLoader_Raw.PABPLoader_Raw
    else:
        raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, and SCAMPS.")
    return this_loader

def train_and_test(config, data_loader_dict):
    """Trains the model."""
    model_trainer = assigntrainer(config.MODEL.NAME)
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)

def test(config, data_loader_dict):
    """Tests the model."""
    model_trainer = assigntrainer(config.MODEL.NAME)
    model_trainer.test(data_loader_dict)

def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method in ["POS", "CHROM", "ICA", "GREEN", "LGI", "PBV", "POS_ENH"]:
            unsupervised_predict(config, data_loader, unsupervised_method)
        else:
            raise ValueError("Not supported unsupervised method!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # get configurations.
    config = get_config(args)
    # ========= assign output folder and complete overriding print() function ===========
    import os
    outfolder = config.LOG.PATH
    outname = "results.txt"

    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    output_file = open(os.path.join(outfolder, outname), 'w')
    _print("output directory: ", outfolder)
    # ================================================================

    # print('Configuration:')
    # print(config, end='\n\n')

    data_loader_dict = dict()
    if config.TOOLBOX_MODE == "only_test":
        test_loader = assignloader(config.TEST.DATA.DATASET)
        if config.TEST.DATA.DATASET is not None and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=config.DATALOADER_WORKERS,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "train_and_test":
        # neural method dataloader
        train_loader = assignloader(config.TRAIN.DATA.DATASET)
        valid_loader = assignloader(config.VALID.DATA.DATASET)
        test_loader = assignloader(config.TEST.DATA.DATASET)

        if config.TRAIN.DATA.DATASET is not None and config.TRAIN.DATA.DATA_PATH:
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=config.DATALOADER_WORKERS,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
        else:
            data_loader_dict['train'] = None

        if config.VALID.DATA.DATASET is not None and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH:
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=config.DATALOADER_WORKERS,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['valid'] = None

        if config.TEST.DATA.DATASET is not None and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=config.DATALOADER_WORKERS,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        unsupervised_loader = assignloader(config.UNSUPERVISED.DATA.DATASET)

        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=config.DATALOADER_WORKERS,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError("Unsupported TOOLBOX_MODE! Currently support train_and_test / only_test / unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')

    output_file.close()
    exit()

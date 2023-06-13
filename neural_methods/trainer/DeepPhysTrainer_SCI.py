"""Trainer for DeepPhys."""

import logging
import os

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.trainer.DeepPhysTrainer_Raw import DeepPhysTrainer
from tqdm import tqdm

from enhancement.model import Finetunemodel

class DeepPhysTrainer(DeepPhysTrainer):

    def __init__(self, config, data_loader):
        '''
        Inherit from PhysnetTrainer_Raw.py, overwrite some functions:
            set_train(): to set the status of both IEM and Physnet model to train()
            set_eval(): to set the status of both IEM and Physnet model to eval()
            load_model_weight(): load both Physnet and IEM, freeze the parameters of Physnet; 
                                Raise error when Physnet is not assigned
            set_train_config(): Set the optimizer and scheduler
        '''
        super().__init__(config, data_loader)


    def set_train(self):
        self.model.train()
        self.enhancemodel.train()

    def set_eval(self):
        self.model.eval()
        self.enhancemodel.eval()

    def load_model_weight(self):        
        # Load the pretrained model if the pretrained model path is assigned
        if self.config.TRAIN.MODEL_PATH != "":
            self.model.load_state_dict(torch.load(self.config.TRAIN.MODEL_PATH, map_location=self.config.DEVICE))
            print("Train IEM using Deepphys: ", self.config.TRAIN.MODEL_PATH)
        else:
            raise ValueError("config.TRAIN.MODEL_PATH is needed for IEM training!")

        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                module.track_running_stats = False
        print("The Deepphys part is freezed!")

        # Load & init enhancement model 
        p = self.config.TRAIN.ENHANCEMODEL_PATH
        if (p is not None) and (p != ""): 
            self.enhancemodel = Finetunemodel(weights=p).to(self.device)
            print("Transfer learning IEM at ", p)
        else:
            self.enhancemodel = Finetunemodel(weights=None).to(self.device)
            print("Train IEM from scratch.")
        
    def set_train_config(self):
        # Change the optimzed parameters to enhancemodel parameters
        params = list(self.enhancemodel.parameters()) + list(self.model.parameters())
        self.optimizer = optim.AdamW(params=params, lr=self.config.TRAIN.LR, weight_decay=0)
        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)

    
    def predictppg(self, feed):
        # Input feed to image enhancement model (IEM)
        permute = [2, 1, 0] # permute RGB to BGR channel
        thisfeed = (feed/255.).to(torch.float32).to(self.device)[:, :, permute, :, :] #4,self.chunk_len,3,72,72
        enh = torch.zeros_like(thisfeed)
        for t in range(self.chunk_len):
            enh[:,t,:,:,:] = self.enhancemodel(thisfeed[:,t,:,:,:])[1] # for [0] it look like whitening masked
        enh = (enh * 255.)[:, :, permute, :, :] # permute back

        diff_feed = self.diff_normalize(enh)
        feed = torch.cat([diff_feed, enh], dim=2) # Concat the diffnormalized data with raw data for DeepPhys Training
        N, D, C, H, W = feed.shape
        feed = feed.view(N * D, C, H, W)
        ppg = self.model(feed)
        return ppg

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.ENHANCEMODEL_PATH):
                raise ValueError("[Only Test] Please check config.INFERENCE.ENHANCEMODEL_PATH!")
            else:
                if self.config.ENHANCEMODEL_NAME == "SCI":
                    self.enhancemodel = Finetunemodel(self.config.INFERENCE.ENHANCEMODEL_PATH)
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("[Path error] Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("[Only Test] Enhance path: ", self.config.INFERENCE.ENHANCEMODEL_PATH)
            print("[Only Test] Deepphys path: ", self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                # for self.model, keep as is
                if self.config.ENHANCEMODEL_NAME == "SCI":
                    self.enhancemodel = Finetunemodel(last_epoch_model_path)
                print("[Test Last Epoch] Enhance path: ", last_epoch_model_path)
                print("[Test Last Epoch] Deepphys path: ", self.config.TRAIN.MODEL_PATH)
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                if self.config.ENHANCEMODEL_NAME == "SCI":
                    self.enhancemodel = Finetunemodel(best_model_path)
                print("[Test Best Epoch] Enhance path: ", best_model_path)
                print("[Test Best Epoch] Deepphys path: ", self.config.TRAIN.MODEL_PATH)
                self.enhancemodel.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.device)
        self.enhancemodel = self.enhancemodel.to(self.device)
        self.set_eval()

        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.device), test_batch[1].to(self.device)
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = self.predictppg(data_test)

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        
        print('')
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.enhancemodel.state_dict(), model_path)
        print('Save Enhance Model Path: ', model_path)

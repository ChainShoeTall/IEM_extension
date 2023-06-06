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
            print("Retrain model at ", self.config.TRAIN.MODEL_PATH)
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


    def diff_normalize(self, data):
        """Quicker implementation of diff_normalize_data() with offset=0, no inner loops"""
        b,n,c,h,w, = data.shape
        res = (data[:, 1:, :, :, : ] - data[:, :-1:, :, :, :]) / ((1e-7 + data[:, 1:, :, :, : ] + data[:, :-1:,:,  :, :]))
        sd = torch.std(res, dim=(1,2,3,4), unbiased=False)
        res = torch.div(res, sd.reshape(b,1,1,1,1))
        res = torch.cat([res, torch.zeros_like(res[:, 0:1, :, :, :])], dim=1)
        return res   

    
    def predictppg(self, feed):
        # Input feed to image enhancement model (IEM)
        permute = [2, 1, 0] # permute RGB to BGR channel
        thisfeed = (feed/255.).to(torch.float32).to(self.device)[:, :, permute, :, :] #4,128,3,72,72
        enh = torch.zeros_like(thisfeed)
        for t in range(128):
            enh[:,t,:,:,:] = self.enhancemodel(thisfeed[:,t,:,:,:])[1] # for [0] it look like whitening masked
        enh = (enh * 255.)[:, :, permute, :, :] # permute back

        diff_feed = self.diff_normalize(feed)
        feed = torch.cat([diff_feed, feed], dim=2) # Concat the diffnormalized data with raw data for DeepPhys Training
        N, D, C, H, W = feed.shape
        feed = feed.view(N * D, C, H, W)
        ppg = self.model(feed)
        return ppg

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.set_train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                labels = labels.view(-1, 1)

                self.optimizer.zero_grad()
                pred_ppg = self.predictppg(data)

                # Need Normalization for MSE?
                # pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)
                # BVP_label = (BVP_label - torch.mean(BVP_label))/torch.std(BVP_label)

                loss = self.criterion(pred_ppg.view(-1,128), labels.view(-1, 128))
                loss.requires_grad_(True) # Otherwise will cause "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                tbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})
            self.save_model(epoch)
            # print(self.model.module.motion_conv1.weight[0,0,0,0].item())
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.set_eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

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
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("[Only Test] Enhance path: ", self.config.INFERENCE.ENHANCEMODEL_PATH)
            print("[Only Test] PhysNet path: ", self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                # for self.model, keep as is
                if self.config.ENHANCEMODEL_NAME == "SCI":
                    self.enhancemodel = Finetunemodel(last_epoch_model_path)
                print("[Test Last Epoch] Enhance path: ", last_epoch_model_path)
                print("[Test Last Epoch] PhysNet path: ", self.config.TRAIN.MODEL_PATH)
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.device)
        self.enhancemodel = self.enhancemodel.to(self.device)
        self.set_eval()

        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.device), test_batch[1].to(self.device)
                # N, D, C, H, W = data_test.shape
                # data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                # pred_ppg_test = self.model(data_test)
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

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

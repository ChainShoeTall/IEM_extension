"""

PhysNet + Self-Calibration Illumination Model Trainer.
This trainer will freeze the parameters of pretrained PhysNet model and 
only train the SCI model to make it adapted to rPPG extraction.

"""
import os

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.PhysnetTrainer_Raw import PhysnetTrainerRaw
# from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm

from enhancement.model import Finetunemodel, DecomNet
from neural_methods.loss.ShiftLoss import ShiftLoss

class PhysnetTrainer(PhysnetTrainerRaw):

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
        # Freeze the model
        # VERY IMPORTANT for keeping same behaviour at test immediately after training and test afterwards
        # https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/18        
        # Load the pretrained model if the pretrained model path is assigned
        if self.config.TRAIN.MODEL_PATH != "":
            self.model.load_state_dict(torch.load(self.config.TRAIN.MODEL_PATH, map_location=self.config.DEVICE))
        else:
            raise ValueError("config.TRAIN.MODEL_PATH is needed for IEM training!")
        
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                module.track_running_stats = False
        print("The PhysNet part is freezed!")
        
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
        self.optimizer = optim.Adam(params=params, lr=self.config.TRAIN.LR)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches
        )
    
    def predictppg(self, feed):
        """
        Overwrite the predictppg func. Add enhancemodel before feed into model
        The SKIP_ENH is replaced by using PhysnetTrainerRaw instead
        """
        # Input feed to image enhancement model (IEM)
        permute = [2, 1, 0] # permute RGB to BGR channel
        thisfeed = (feed/255.).to(torch.float32).to(self.device)[:, permute, :, :, :] #4,3,128,72,72
        enh = torch.zeros_like(thisfeed)
        for t in range(128):
            enh[:,:,t,:,:] = self.enhancemodel(thisfeed[:,:,t,:,:])[1] # for [0] it look like whitening masked
        enh = (enh * 255.)[:, permute, :, :, :] # permute back

        enh = self.diff_normalize(enh)
        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(enh)
        return rPPG

    def test(self, data_loader):
        """ Runs the model on test sets."""
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
                raise ValueError("[Path error] Please check config.INFERENCE.MODEL_PATH!")
            self.model = PhysNet_padding_Encoder_Decoder_MAX(
                frames=self.config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("[Only Test] Enhance path: ", self.config.INFERENCE.ENHANCEMODEL_PATH)
            print("[Only Test] PhysNet path: ", self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                # changed, insert '_enhancement' in file name
                last_epoch_model_path = os.path.join(
                self.config.MODEL.MODEL_DIR, self.config.TRAIN.MODEL_FILE_NAME + '_enhancement_' + self.config.ENHANCEMODEL_NAME +'_Epoch' + str(self.config.TRAIN.EPOCHS - 1) + '.pth')
                
                # for self.model, keep as is
                if self.config.ENHANCEMODEL_NAME == "SCI":
                    self.enhancemodel = Finetunemodel(last_epoch_model_path)
                print("[Test Last Epoch] Enhance path: ", self.config.INFERENCE.ENHANCEMODEL_PATH)
                print("[Test Last Epoch] PhysNet path: ", last_epoch_model_path)
            else:
                # raise Exception("not implemented for now...")
                best_model_path = os.path.join(
                    self.config.MODEL.MODEL_DIR, self.config.TRAIN.MODEL_FILE_NAME + '_enhancement_' + self.config.ENHANCEMODEL_NAME + '_Epoch' + str(self.best_epoch) + '.pth')
                if self.config.ENHANCEMODEL_NAME == "SCI":
                    self.enhancemodel = Finetunemodel(best_model_path)
                print("[Test Best Epoch] Enhance path: ", self.config.INFERENCE.ENHANCEMODEL_PATH)
                print("[Test Best Epoch] PhysNet path: ", best_model_path)

        self.enhancemodel = self.enhancemodel.to(self.config.DEVICE)
        self.model = self.model.to(self.config.DEVICE)
        self.set_eval()

        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader['test'])):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                pred_ppg_test = self.predictppg(data)

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    # Create the key to store the result of sample: subj_index
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        '''changed to save the enhancement only!'''
        if not os.path.exists(self.config.MODEL.MODEL_DIR):
            os.makedirs(self.config.MODEL.MODEL_DIR)

        # changed these 2 lines, to store the enhancement model only
        model_path = os.path.join(
            self.config.MODEL.MODEL_DIR, self.config.TRAIN.MODEL_FILE_NAME + '_enhancement_' + self.config.ENHANCEMODEL_NAME + '_Epoch' + str(index) + '.pth')
        torch.save(self.enhancemodel.state_dict(), model_path)

        print('Saved Model Path: ', model_path)

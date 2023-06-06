"""PhysNet Trainer.

tensor input: raw image BGR color space with [0.255], shape in [N,C,T,H,W]
    N: Number of frame clips;
    C: Color channel;
    T: Number of frames in each clip;
    H, W: Height and width of the frame

"""
import os

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm

from neural_methods.loss.ShiftLoss import ShiftLoss

class PhysnetTrainerRaw(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.config = config
        self.num_train_batches = len(data_loader["train"]) if "train" in data_loader else 1
        self.min_valid_loss = None
        self.best_epoch = 0
        self.model = PhysNet_padding_Encoder_Decoder_MAX(frames=self.config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if self.config.TOOLBOX_MODE == "train_and_test":        
            self.load_model_weight()
            self.set_loss_func()
            self.set_train_config()

    def diff_normalize(self, data):
        """Quicker implementation of diff_normalize_data() with offset=0, no inner loops"""
        b,c,n,h,w, = data.shape
        res = (data[:, :, 1:, :, : ] - data[:, :, :-1:, :, :]) / ((1e-7 + data[:, :, 1:, :, : ] + data[:, :, :-1:, :, :]))
        sd = torch.std(res, dim=(1,2,3,4), unbiased=False)
        res = torch.div(res, sd.reshape(b,1,1,1,1))
        res = torch.cat([res, torch.zeros_like(res[:, :, 0:1, :, :])], dim=2)
        return res    
     
    def predictppg(self, feed):
        """
        Predict the rPPG value with the model
        Input: the original batch which is in BGR color space within [0, 255], shape NCTHW
        Output: rPPG signal (shape in N,T)
        """
        enh = self.diff_normalize(feed)
        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(enh)
        return rPPG
    
    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def set_loss_func(self):
        # Use Negative Pearson Correlation coefficients as training loss
        if self.config.TRAIN.LOSS == "NP":
            self.loss_model = Neg_Pearson()
            print('NP loss used')
        # Use Temporal-Shifted Negative Pearson Correlation coefficients as training loss
        elif self.config.TRAIN.LOSS == "SNP":
            self.loss_model = ShiftLoss(fn=Neg_Pearson(), shift=int(self.config.TRAIN.DATA.FS/3))
            print('SNP loss used')
        else: 
            raise ValueError("wrong loss function specified!")
        
    def load_model_weight(self):        
        # Load the pretrained model if the pretrained model path is assigned
        if self.config.TRAIN.MODEL_PATH != "":
            self.model.load_state_dict(torch.load(self.config.TRAIN.MODEL_PATH, map_location=self.config.DEVICE))
        else:
            print("Training from scratch!")
        
    def set_train_config(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.TRAIN.LR)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches
        )

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.config.TRAIN.EPOCHS):
            print('')
            print(f"========Training Epoch: {epoch}========")
            running_loss = 0.0
            train_loss = []

            # train() will enable Batch Normalization and Dropout (if any)
            self.set_train()

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # Pass the input throught the neural network
                feed = batch[0].to(torch.float32).to(self.device)
                BVP_label = batch[1].to(torch.float32).to(self.device)
                rPPG = self.predictppg(feed)

                # Normalize both predicted rPPG and BVP label for loss calculation
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
                BVP_label = (BVP_label - torch.mean(BVP_label))/torch.std(BVP_label)

                # Compute training loss
                loss = self.loss_model(rPPG, BVP_label)

                # Backward propogation
                loss.backward()

                # To compute average running loss
                running_loss += loss.item()
                if idx % 100 == 99:  # print running loss every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)

            # No validation if TEST.USE_LAST_EPOCH is True
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
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ========Validing=======")
        valid_loss = []

        # eval() will disable Batch Normalization and Dropout (if any), but still compute gradient
        self.set_eval()

        # torch.no_grad() save gpu memory and faster
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for _, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                feed = valid_batch[0].to(torch.float32).to(self.device)
                rPPG = self.predictppg(feed)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label))/torch.std(BVP_label)  # normalize
                
                # Just compute the loss
                loss_ecg = self.loss_model(rPPG, BVP_label)

                valid_loss.append(loss_ecg.item())
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("=======Testing=======")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("[Only Test] PhysNet path: ", self.config.INFERENCE.MODEL_PATH) 
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.config.MODEL.MODEL_DIR, self.config.TRAIN.MODEL_FILE_NAME + '_Epoch' + str(self.config.TRAIN.EPOCHS - 1) + '.pth'
                )
                print("[Test Last Epoch] PhysNet path: ", last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.config.MODEL.MODEL_DIR, self.config.TRAIN.MODEL_FILE_NAME + '_Epoch' + str(self.best_epoch) + '.pth')
                print("[Test Best Epoch] PhysNet path: ", best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader['test'])):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.predictppg(data)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.config.MODEL.MODEL_DIR):
            os.makedirs(self.config.MODEL.MODEL_DIR)
        model_path = os.path.join(
            self.config.MODEL.MODEL_DIR, self.config.TRAIN.MODEL_FILE_NAME + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

"""PhysNet-Trad-E Trainer."""
import os
import cv2

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm

from enhancement.model import Finetunemodel
from neural_methods.loss.ShiftLoss import ShiftLoss

class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()

        self.ie_module = config.IEM

        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS

        # batch number is only needed for training
        self.num_train_batches = len(data_loader["train"]) if "train" in data_loader else 1

        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0     

         # assign loss function
        if self.config.TRAIN.LOSS == "NP":
            self.loss_model = Neg_Pearson()
            print('NP loss used')
        elif self.config.TRAIN.LOSS == "SNP":
            self.loss_model = ShiftLoss(fn=Neg_Pearson(), shift=int(self.config.TRAIN.DATA.FS/3))
            print('SNP loss used')
        else: 
            raise ValueError("wrong loss function specified!")

    def gamma_correction(self, input):
        return torch.pow(input, 1.0/self.config.TRAD.GAMMA_VALUE)

    def histogram_eq(self, input):
        output = np.zeros_like(input)
        for ibatch in range(input.shape[0]):
            for ich in range(input.shape[1]):
                for t in range(input.shape[2]):
                    output[ibatch, ich, t] = cv2.equalizeHist(input[ibatch, ich, t])
        return output

    def predictppg(self, feed):
        """feed the original batch which is in BGR color space with scale [0,255], shape N,C,T,H,W
        output rppg, shape N,T
        """
        if np.abs(self.config.GAMMA_VALUE - 0.0) > 0.01: # No gamma correction when gamma value = 0
            if self.config.TRAD.NAME == "GC":
                thisfeed = (feed/255.0).to(torch.float32).to(self.device) #4,3,128,72,72
                enh = self.gamma_correction(thisfeed)*255.0
            elif self.config.TRAD.NAME == "HE":
                thisfeed = (feed).cpu().numpy() #4,3,128,72,72
                enh = torch.from_numpy(self.histogram_eq(thisfeed.astype(np.uint8))).to(torch.float32).to(self.device)
        else:
            enh = feed

        enh = self.diff_normalize_quick(enh)

        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(enh)
        return rPPG

    def diff_normalize_quick(self, data):
        """quicker implementation of diff_normalize_data() with offset=0, no inner loops"""
        b,c,n,h,w, = data.shape
        res = (data[:, :, 1:, :, : ] - data[:, :, :-1:, :, :]) / ((1e-7 + data[:, :, 1:, :, : ] + data[:, :, :-1:, :, :]))
        sd = torch.std(res, dim=(1,2,3,4), unbiased=False)
        res = torch.div(res, sd.reshape(b,1,1,1,1))
        res = torch.cat([res, torch.zeros_like(res[:, :, 0:1, :, :])], dim=2)
        return res

    def diff_normalize_data(self, data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        """rewrite for our torch tensors in batch"""    
        b,c,n,h,w, = data.shape
        difflen = n-1
        
        res = list()
        for i in range(b):
            diffs = list()
            # in implementaion of rppg-toolbox, offset=1, but it can be 0
            offset = 0
            for j in range(difflen-offset):
                temp = (data[i, :, j+1, :, :] - data[i, :, j, :, :]) / (data[i, :, j+1, :, :] + data[i, :, j, :, :] + 1e-7)
                diffs.append(temp)
            diffbatch = torch.stack(diffs, dim=1)
            diffbatch /= torch.std(diffbatch, unbiased=False)
            diffbatch = torch.cat([diffbatch, torch.zeros_like(diffbatch[:, 0:offset+1, :, :])], dim=1)
            res.append(diffbatch)
        res = torch.stack(res, dim=0)
        return res

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        
        # load my PhysNet here, and freeze it
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=self.config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        p = self.config.TRAIN.MODEL_PATH
        if (p is not None) and (p != ""): 
            self.model.load_state_dict(torch.load(p))
            print("Loaded PhysNet path: ", p)
        else:
            raise Exception("Please provide trained PhysNet weights to freeze...")

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.TRAIN.LR)
        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)    

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                # predict here
                rPPG = self.predictppg(batch[0])

                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss = self.loss_model(rPPG, BVP_label)

                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
            
            self.save_model(epoch)
            
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

            # step that loss if available
            try:
                self.loss_model.step()
            except:
                pass
            
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        # raise Exception("not implemented")
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                
                rPPG = self.predictppg(valid_batch[0])

                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            # if self.config.INFERENCE.SKIP_ENHANCEMENT:
            #     self.enhancemodel = torch.nn.Identity()
            #     print("Enhancement model skipped!")
            # elif not os.path.exists(self.config.INFERENCE.ENHANCEMODEL_PATH):
            #     raise ValueError("Inference enhancement model path error! Please check your yaml.")
            # else:
            #     self.enhancemodel = Finetunemodel(self.config.INFERENCE.ENHANCEMODEL_PATH)
            #     print("Testing uses pretrained enhancement model!")
            #     print(self.config.INFERENCE.ENHANCEMODEL_PATH)

            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference PhysNet model path error! Please check your yaml.")
            self.model = PhysNet_padding_Encoder_Decoder_MAX(
                frames=self.config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained PhysNet model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                # changed, insert '_enhancement' in file name
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_' + self.ie_module + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                # for self.model, keep as is
                self.model.load_state_dict(torch.load(last_epoch_model_path, self.device))
                print("Testing uses last epoch as non-pretrained enhancement model!")
                print(last_epoch_model_path)
            else:
                # raise Exception("not implemented for now...")
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_' + self.ie_module + '_Epoch' + str(self.best_epoch) + '.pth')
                self.model.load_state_dict(torch.load(last_epoch_model_path, self.device))
                print("Testing uses best epoch selected using model selection as non-pretrained enhancement model!")
                print(best_model_path)

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
        '''changed to save the enhancement only!'''
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # changed these 2 lines, to store the enhancement model only
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_' + self.ie_module + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Trad Enhance Model Path: ', model_path)

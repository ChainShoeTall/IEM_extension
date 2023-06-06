"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""

import logging
import os

import numpy as np
import torch
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from enhancement.model import Finetunemodel

from tqdm import tqdm


def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    sbar = tqdm(data_loader["unsupervised"], ncols=80)

    # Load IEM for POS_ENH
    if method_name == "POS_ENH":
        # load Finetunmodel manually when "cuda:0" is not available
        thisdevice = "cuda:0" if torch.cuda.is_available() else "cpu"
        FTModel = Finetunemodel(None).to(thisdevice)
        base_weights = torch.load(config.INFERENCE.ENHANCEMODEL_PATH, map_location=thisdevice)
        pretrained_dict = base_weights
        model_dict = FTModel.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        FTModel.load_state_dict(model_dict)

    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            elif method_name == "POS_ENH":
                permutation = [2, 1, 0]
                thisfeed = data_input.transpose((3,0,1,2))
                thisfeed = torch.tensor(thisfeed/255.).to(torch.float32).to(thisdevice)[permutation, :, :, :] #3,128,72,72
                thisfeed = thisfeed.unsqueeze(0)
                enh = torch.zeros_like(thisfeed)
                for t in range(thisfeed.shape[2]):
                    enh[:,:,t,:,:] = FTModel(thisfeed[:,:,t,:,:])[1] # for [0] it look like whitening masked
                enh = (enh * 255.)[:, permutation, :, :, :] # permute back
                enh = enh.squeeze().permute(1,2,3,0).detach().cpu().numpy()
                BVP = POS_WANG(enh, config.UNSUPERVISED.DATA.FS)
            else:
                raise ValueError("unsupervised method name wrong!")

            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr, pre_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                predict_hr_peak_all.append(pre_hr)
                gt_hr_peak_all.append(gt_hr)
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_fft_hr, pre_fft_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                   fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                predict_hr_fft_all.append(pre_fft_hr)
                gt_hr_fft_all.append(gt_fft_hr)
    print("Used Unsupervised Method: " + method_name)
    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(
                    np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Wrong Test Metric Type")
    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_PEAK))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(
                    np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_PEAK))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_PEAK))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson  (FFT Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Wrong Test Metric Type")

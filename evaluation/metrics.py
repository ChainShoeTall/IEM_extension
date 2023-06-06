import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from matplotlib import pyplot as plt
import os
from utils.SL5Fold import getFilename

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    collected_gt = list()
    collected_pred = list()
    keylist = list(predictions.keys()) # list of all the samples
    for index in keylist:
        prediction = _reform_data_from_dict(predictions[index]) # Sort chunks for each sample
        label = _reform_data_from_dict(labels[index])

        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        
        hr_gt, hr_pred = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method=config.INFERENCE.EVALUATION_METHOD
        )
        collected_pred.append(hr_pred)
        collected_gt.append(hr_gt)

    collected_gt = np.array(collected_gt)
    collected_pred = np.array(collected_pred)
    plt.plot(collected_gt, collected_pred, '.')
    # best linear fit
    d1, d0 = np.polyfit(collected_gt, collected_pred, deg=1) # coeff of 1st and 0th degree
    xseq = np.linspace(np.min(collected_gt), np.max(collected_gt), num=3)
    plt.plot(xseq, d1*xseq + d0)
    # correlation coeffcient
    ceff = np.corrcoef(collected_gt, collected_pred)[0,1]
    plt.title('predicted HR against gt HR, ' + f'correlation coeff = {ceff:.4}')
    plt.xlabel('gt HR/bpm')
    plt.ylabel('predicted HR/bpm')
    plt.savefig(os.path.join(config.LOG.PATH, "hrplot.png"))
    print("Figure saved!")
    #csv
    df = pd.DataFrame(list(zip(keylist, collected_gt, collected_pred)), columns=['index', 'gt', 'pred'])
    df.to_csv(os.path.join(config.LOG.PATH, "hrpredictions.csv"))
    print("HR predictions saved!")

    for metric in config.TEST.METRICS:
        if metric == "MAE":
            MAE_FFT = np.mean(np.abs(collected_pred - collected_gt))
            print("MAE (FFT Label):{0}".format(MAE_FFT))

        elif metric == "RMSE":
            RMSE_FFT = np.sqrt(np.mean(np.square(collected_pred - collected_gt)))
            print("RMSE (FFT Label):{0}".format(RMSE_FFT))

        elif metric == "MAPE":
            MAPE_FFT = np.mean(np.abs((collected_pred - collected_gt) / collected_gt)) * 100
            print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))

        elif metric == "Pearson":
            Pearson_FFT = np.corrcoef(collected_pred, collected_gt)
            print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))

        else:
            raise ValueError("Wrong Test Metric Type")
    # np_save_name = config.TEST.SAVE_RESULT_NAME
    # np.savez(np_save_name, MAE_FFT=MAE_FFT, RMSE_FFT=RMSE_FFT, MAPE_FFT=MAPE_FFT, Pearson_FFT=Pearson_FFT,
    #          predict_hr_fft_all=predict_hr_fft_all, gt_hr_fft_all=gt_hr_fft_all)

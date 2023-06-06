"""The dataloader for UBFC datasets.

Details for the UBFC-RPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import pandas as pd
from datetime import datetime


class PABPLoader_Raw(BaseLoader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC dataset)."""
        print("ET-{:d}_Gain{:d}_P*".format(self.pabp_et, self.pabp_gain))
        data_dirs = glob.glob(data_path + os.sep + "ET-{:d}_Gain{:d}_P*".format(self.pabp_et, self.pabp_gain))
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'Raw/ET-(?:\d)_Gain(?:\d)_P(\d+)', data_dir).group(1), "path": data_dir} for data_dir in data_dirs]
        print(dirs)
        return dirs

    # def split_raw_data(self, data_dirs, begin, end):
    #     """Returns a subset of data dirs, split with begin and end values."""
    #     if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
    #         return data_dirs

    #     file_num = len(data_dirs)
    #     choose_range = range(int(begin * file_num), int(end * file_num))
    #     data_dirs_new = []

    #     for i in choose_range:
    #         data_dirs_new.append(data_dirs[i])

    #     return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        frame_fn = os.path.join(data_dirs[i]['path'],filename + "_c270.avi")
        frame_time_fn = os.path.join(data_dirs[i]['path'],filename + "_c270_frames.csv")
        bvps_fn = os.path.join(data_dirs[i]['path'],"../../brght25", filename + "_c270.npz")

        # ================================ #
        frame_time = pd.read_csv(frame_time_fn)
        VidObj = cv2.VideoCapture(frame_fn)

        frame_start = frame_time["times"][0]
        frame_end = frame_time["times"][frame_time.shape[0]-1]
        frame_start = datetime.strptime(frame_start, "%d/%m/%Y, %H:%M:%S.%f")
        frame_end = datetime.strptime(frame_end, "%d/%m/%Y, %H:%M:%S.%f")

        labels  = np.load(bvps_fn, allow_pickle=True)
        ppg = labels["arr_0"].item()['gt_PPG']
        ppg_start = ppg[0][0]
        ppg_end = ppg[-1][0]
        time_start = max(ppg_start, frame_start)
        time_end = min(ppg_end, frame_end)

        frames = list()
        for i,i_frame_time in enumerate(frame_time["times"]):
            time_now = datetime.strptime(i_frame_time, "%d/%m/%Y, %H:%M:%S.%f")
            if time_now > time_start and time_now < time_end:
                VidObj.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = VidObj.read()
                if success:
                    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                    frame = np.asarray(frame)
                    frames.append(frame)
        VidObj.release()
        frames = np.asarray(frames)

        bvps = list()
        for i, i_ppg_time in enumerate(ppg):
            time_now = i_ppg_time[0]
            if time_now > time_start and time_now < time_end:
                bvps.append(i_ppg_time[1])
        bvps = np.asarray(bvps)
        # ================================ #
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        labels = np.load(bvp_file, allow_pickle=True)
        df = labels["arr_0"].item()
        bvp = list()
        for i_df in df["gt_PPG"]:
            bvp.append(i_df[1])
        return np.asarray(bvp)

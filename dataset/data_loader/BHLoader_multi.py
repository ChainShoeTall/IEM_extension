"""The dataloader for BH-rPPG datasets.

Please insert the correct citation here
"""
import glob
import os
import re

import cv2
import h5py
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class BHLoader(BaseLoader):
    """The data loader for the BH-rPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an BH-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |  |-- 0_0/
                     |  |   |-- 0_0/
                     |  |   |   |-- Frame_00000.png
                     |  |   |   |-- Frame_00001.png
                     |  |   |   |-- ...
                     |  |   |   |-- Frame_?????.png
                     |  |   |-- sensor.csv
                     |  |   |-- timestamp.csv
                     |  |   |-- wave.csv
                     |  |-- 0_1/
                     |  |-- 0_2/
                     |  |-- 1_0/
                     |...
                     |...
                     |  |-- 34_2/
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path.
        adopt to low/middle/high settings
        """
        # sorted or not sorted may make differences...
        # data_dirs = glob.glob(data_path + os.sep + "*_*")
        data_dirs = sorted(glob.glob(data_path + os.sep + "*_*"))
        
        if not data_dirs:
            raise ValueError("data paths empty!")

        # my addition ifelse block
        dirs = list()
        if self.specific == 'low':
            for data_dir in data_dirs:
                matches = re.findall("Pub_BH-rPPG_FULL/(\d|\d\d)_0", data_dir)
                if len(matches) > 0:
                    dirs.append({"index": (matches[0], '0'), "path": data_dir} )
        elif self.specific == 'middle':
            for data_dir in data_dirs:
                matches = re.findall("Pub_BH-rPPG_FULL/(\d|\d\d)_1", data_dir)
                if len(matches) > 0:
                    dirs.append({"index": (matches[0], '1'), "path": data_dir} )
        elif self.specific == 'high':
            for data_dir in data_dirs:
                matches = re.findall("Pub_BH-rPPG_FULL/(\d|\d\d)_2", data_dir)
                if len(matches) > 0:
                    dirs.append({"index": (matches[0], '2'), "path": data_dir} )
        elif self.specific == 'default':
            dirs = [{"index": re.search("Pub_BH-rPPG_FULL/(\d|\d\d)_(\d)", data_dir).group(1,2),
            "path": data_dir} for data_dir in data_dirs]
        else:
            raise Exception("this specific requitement not supported")

        return dirs
    

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        '''for parallel computing'''
        saved_filename = data_dirs[i]["index"][0] + "_" + data_dirs[i]["index"][1]

        frames = self.read_video(os.path.join(data_dirs[i]["path"], saved_filename))
        bvps = self.read_wave(os.path.join(data_dirs[i]["path"],  "wave.csv"))
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(imgs_dir):
        """Reads a video file, returns frames(T,H,W,3) 
        imgs_dir doesn't include os.sep"""
        max_img_num = 99999
        frames = list()
        for i in range(max_img_num):
            img_path = os.path.join(imgs_dir, r"Frame_{:05d}.png".format(i))
            if os.path.exists(img_path):
                f1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
                f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                frames.append(f1)
            else:
                break
        frame_arr = np.stack(frames, axis=0)
        return frame_arr

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[1:-1]]
        return np.asarray(bvp)
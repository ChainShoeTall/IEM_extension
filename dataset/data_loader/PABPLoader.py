"""The dataloader for PABP-rPPG datasets.

Please insert the correct citation here
"""
import glob
import os
import re

import cv2
import h5py
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader


class PABPLoader(BaseLoader):
    """The data loader for the BH-rPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an BH-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                RawData/
                    |
                    id + {\d+} + {ipad/iphone} + _L + {illum} + _D + {CameraLen} + _recording + {\d}
                    |
                    |-- [video_id + {\d+} + _ipad_L + {illum} + _D + {CameraLen} + _recording + {\d} + .npy
                    |
                    |-- ground-truth_id + {\d+} + {ipad/iphone} + _L + {illum} + _D + {CameraLen} + _recording + {\d} + .pkl
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

        glob_path = "{}/id*_{}_L{}_D0.{}_recording*".format(data_path, 
            self.pabp_device, str(self.pabp_illumination), str(self.pabp_distance))
        data_dirs = sorted(glob.glob(glob_path))
        
        if not data_dirs:
            raise ValueError("data paths empty!")

        # my addition ifelse block
        dirs = list()

        dirs = [{"index": re.search(f"id(\d+)_(?:ipad|iphone)_L(?:\d+)_D0.(?:\d)_recording(\d+)", data_dir).group(1),
                "path": data_dir} for data_dir in data_dirs]
        # print(dirs)

        return dirs
    
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        
        ######### to be modified
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])
        #########

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        '''for parallel computing'''
        saved_filename = data_dirs[i]["path"].split(os.sep)[-1]

        frames = np.load(os.path.join(data_dirs[i]["path"], "video_" + saved_filename + ".npy"))
        frames = frames[:,:,:,::-1]
        bvps = self.read_wave(os.path.join(data_dirs[i]["path"],  "ground-truth_" + saved_filename + ".pkl"))
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        frames_clips, bvps_clips = self.preprocess_without_facecrop(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def preprocess_without_facecrop(self, frames, bvps, config_preprocess):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        Returns:
            frame_clips(np.array): processed video data by frames
            bvps_clips(np.array): processed bvp (ppg) labels by frames
        """
        # Check data transformation type
        data = list()  # Video data
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, bvps_clips = self.chunk(
                data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    @staticmethod
    def read_video(imgs_dir):
        """Reads a video file, returns frames(T,H,W,3) 
        imgs_dir doesn't include os.sep"""
        pass

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        pkls = pd.read_pickle(bvp_file)
        return pkls["ppg_data"]

import os
import numpy as np

'''Not used anymore'''

def getFilename(config, folds=None):
    five_fold_dir = "./five_folds"
    if not os.path.exists(five_fold_dir):
        os.mkdir(five_fold_dir)
    if config["TEST"]["DATA"]["DATASET"] == "PABP":
        if config["MODEL"]["NAME"] == "Physnet-E":
            np_save_name = "_".join([config["TEST"]["DATA"]["DATASET"],
                    config["TEST"]["DATA"]["SPECIFIC"],
                    "TestOnDistance",
                    str(int(config["TEST"]["DATA"]["PABP_DISTANCE"])),
                    config["MODEL"]["NAME"],
                    (str(int(config["INFERENCE"]["TEST_FOLD"]*10)) if folds is None else str(int(folds*10)))
                ])
        else:
            folds = None
            np_save_name = "_".join([config["TEST"]["DATA"]["DATASET"],
                    config["TEST"]["DATA"]["SPECIFIC"],
                    "TestOnDistance",
                    str(int(config["TEST"]["DATA"]["PABP_DISTANCE"])),
                    config["MODEL"]["NAME"],
                    (str(int(config["INFERENCE"]["TEST_FOLD"]*10)) if folds is None else str(int(folds*10)))
                ])
    else:
        if "TEST_FOLD" in config["INFERENCE"].keys(): # For inference on BH-high, no info about which model used for test
            if config["INFERENCE"]["TEST_FOLD"] is None:
                np_save_name = "_".join([config["TEST"]["DATA"]["DATASET"],
                    config["TEST"]["DATA"]["SPECIFIC"],
                    "TestOn",
                    (str(int(config["TEST"]["DATA"]["BEGIN"]*10)) if folds is None else str(int(folds*10))),
                    config["MODEL"]["NAME"]
                ])
            else:
                np_save_name = "_".join([config["TEST"]["DATA"]["DATASET"],
                config["TEST"]["DATA"]["SPECIFIC"],
                "TestOn",
                str(int(config["INFERENCE"]["TEST_FOLD"]*10)),
                config["MODEL"]["NAME"]
            ])
        else:
            np_save_name = "_".join([config["TEST"]["DATA"]["DATASET"],
                    config["TEST"]["DATA"]["SPECIFIC"],
                    "TestOn",
                    (str(int(config["TEST"]["DATA"]["BEGIN"]*10)) if folds is None else str(int(folds*10))),
                    config["MODEL"]["NAME"]
                ])
    return os.path.join("./five_folds", np_save_name + ".npz")

def load5FoldNpz(config, five_folds = [0.0, 0.5]):
    # five_folds = [0.0, 0.2, 0.4, 0.6, 0.8]
    result_dict = {"MAE_FFT": list(),
                   "RMSE_FFT": list(),
                   "Pearson_FFT": list()}
    if config["MODEL"]["NAME"] == "Physnet-T":
        five_folds = [int(config["GAMMA_VALUE"]*10)]
    for five_fold in five_folds:
        npz_path = getFilename(config, five_fold)
        data = np.load(npz_path)
        result_dict["MAE_FFT"].append(data["MAE_FFT"].item())
        result_dict["RMSE_FFT"].append(data["RMSE_FFT"].item())
        result_dict["Pearson_FFT"].append(data["Pearson_FFT"][0,1])

    # if config["MODEL"]["NAME"] == "Physnet-T":
    #     np_save_name = "_".join([config["TEST"]["DATA"]["DATASET"],
    #     config["TEST"]["DATA"]["SPECIFIC"],
    #     "TestOn",
    #     str(int(config["INFERENCE"]["TEST_FOLD"]*10)),
    #     config["MODEL"]["NAME"]
    #     ])
    return result_dict

def loadNpz(npz_path, key_list):
    result_dict = {key: list() for key in key_list}
    data = np.load(npz_path)
    result_dict["MAE_FFT"].append(data["MAE_FFT"].item())
    result_dict["RMSE_FFT"].append(data["RMSE_FFT"].item())
    result_dict["Pearson_FFT"].append(data["Pearson_FFT"][0,1])
    return result_dict
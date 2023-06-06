import os
import yaml

def train_bhlow_2fold(model_path, test_except): 
    yaml_path = "SharedTrainedModel/yamls/test_phys-sci_bhlow.yaml"
    with open(yaml_path, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    config_data["TOOLBOX_MODE"] = "train_and_test"

    config_data["TRAIN"]["MODEL_PATH"] = model_path
    config_data["TRAIN"]["DATA"]["BEGIN"] = 0.0
    config_data["TRAIN"]["DATA"]["END"] = 0.5
    config_data["TRAIN"]["DATA"]["EXCEPT"] = not test_except

    config_data["VALID"]["DATA"]["DATASET"] = "BH"

    config_data["TEST"]["DATA"]["BEGIN"] = 0.0
    config_data["TEST"]["DATA"]["END"] = 0.5
    config_data["TEST"]["DATA"]["EXCEPT"] = test_except
    config_data["TEST"]["USE_LAST_EPOCH"] = True
    with open(yaml_path, 'w', encoding='utf-8') as nf:
        yaml.dump(config_data, nf, sort_keys=False)

    os.system("python main.py --config_file " + yaml_path)

    return config_data

def test_bhlow_2fold(model_path, enhancemodel_path, test_except): 
    yaml_path = "SharedTrainedModel/yamls/test_phys-sci_bhlow.yaml"
    with open(yaml_path, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    config_data["TOOLBOX_MODE"] = "only_test"
    config_data["TEST"]["DATA"]["BEGIN"] = 0.0
    config_data["TEST"]["DATA"]["END"] = 0.5
    config_data["TEST"]["DATA"]["EXCEPT"] = test_except
    config_data["INFERENCE"]["MODEL_PATH"] = model_path
    config_data["INFERENCE"]["ENHANCEMODEL_PATH"] = enhancemodel_path

    with open(yaml_path, 'w', encoding='utf-8') as nf:
        yaml.dump(config_data, nf, sort_keys=False)

    os.system("python main.py --config_file " + yaml_path)

    return config_data

if __name__ == "__main__":

    '''To produce results on Feb for IEM+PhysNet'''
    # model_path = "production/models/Feb_results/Feb_ubfc_physnet_NPLoss.pth"
    # enh_path_part1 = "production/models/Feb_results/Feb_SCI_bhlow_teston_0.0_0.5.pth"
    # enh_path_part2 = "production/models/Feb_results/Feb_SCI_bhlow_teston_0.5_1.0.pthh"
    # test_bhlow_2fold(model_path, enh_path_part1, True)
    # test_bhlow_2fold(model_path, enh_path_part2, False)

    '''Try to train new model to reproduce the result'''
    model_path = "production/models/ubfc_physnet_NPLoss.pth"
    # enh_path_part1 = "production/models/SCI_bhlow_another_teston_0.0_0.5.pth"
    # enh_path_part2 = "production/models/SCI_bhlow_another_teston_0.5_1.0.pth"
    train_bhlow_2fold(model_path, False)




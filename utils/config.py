import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    work_path = 'C:\\Users\\User\\OneDrive - RMIT University\\experiments\\Multivariate_experiments\\DL_model_with_maps_data'

    config.callbacks.tensorboard_log_dir = os.path.join(work_path,"experiments2", "logs"  )
    #config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("run_%Y_%m_%d-%H_%M_%S",time.localtime()), config.exp.name, "checkpoints/")
    config.callbacks.checkpoint_dir = os.path.join(work_path,"experiments2", time.strftime("run_%Y_%m_%d-%H_%M",time.localtime()), config.exp.name, "checkpoints")
    config.callbacks.checkpoint_dir_autenc = os.path.join(work_path,"experiments2", time.strftime("run_%Y_%m_%d-%H_%M",time.localtime()), config.exp.name+'_auc', "checkpoints")
     

    return config

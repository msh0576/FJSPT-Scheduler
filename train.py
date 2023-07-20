import copy
import json
import os
import random
import time
from collections import deque
import logging

import gym
import pandas as pd
import torch
import numpy as np
from hgnn_models.TFJSPTrainer_hgnn import TFJSPTrainer_hgnn
# from visdom import Visdom
from options import get_options


from utils.utils import create_logger, copy_all_src

from DHJS_models.TFJSPTrainer_dhjs import TFJSPTrainer_DHJS
from matnet_models.TFJSPTrainer_matnet import TFJSPTrainer_matnet
from DTrans_models.TFJSPTrainer_dtrans import TFJSPTrainer_DTrans
from GTrans_models.TFJSPTrainer_gtrans import TFJSPTrainer_GTrans

env_paras = {}
model_paras = {}
train_paras = {}
test_paras = {}
optimizer_paras = {}
logger_paras = {}



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(opts):
    # === load config ===
    with open("./configs/config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    change_paras = load_dict["change_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    optimizer_paras = load_dict["optimizer_paras"]
    logger_paras = load_dict["logger_paras"]
    test_paras = load_dict["test_paras"]
    
    # === modify config ===    
    model_paras['sqrt_embedding_dim'] = model_paras['embedding_dim']**(1/2)
    model_paras['sqrt_qkv_dim'] = model_paras['qkv_dim']**(1/2)
    model_paras['ms_layer1_init'] = (1/2)**(1/2)
    model_paras['ms_layer2_init'] = (1/16)**(1/2)
    
    logger_paras['log_file']['desc'] = opts.log_file_desc
    
    return load_dict, env_paras, change_paras, model_paras, train_paras, test_paras,\
        optimizer_paras, logger_paras



def main(opts):
    # ========= Pytorch initialization ==========
    load_dict, env_paras, change_paras, model_paras, train_paras, test_paras,\
        optimizer_paras, logger_paras = load_config(opts)
    
    # ===
    setup_seed(seed=1)
    create_logger(**logger_paras)
    _print_config(load_dict)
    
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    device = torch.device("cuda:"+str(opts.cuda) if torch.cuda.is_available() else "cpu")
    # opts.device = device
    
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    
    # ===
    env_paras["device"] = device
    test_paras["device"] = device
    model_paras["device"] = device
    env_paras["dynamic"]['max_ope_per_job'] = env_paras['num_opes'] // env_paras['num_mas']
    
    model_paras["batch_size"] = env_paras["batch_size"]
    model_paras["proctime_per_ope_max"] = env_paras["proctime_per_ope_max"]
    model_paras["transtime_btw_ma_max"] = env_paras["transtime_btw_ma_max"]
    
    model_paras["checkpoint_encoder"] = opts.checkpoint_encoder
    model_paras["algorithm"] = opts.algorithm
    model_paras['batch_size'] = env_paras['batch_size']
    
    change_paras['enable'] = opts.enable_change_paras
    
    if opts.static:
        env_paras['dynamic'] = None
    if opts.metarl:
        train_paras['meta_rl']['enable'] = True
        if opts.metarl_subgraphs:
            train_paras['meta_rl']['use_subgraphs'] = True
        env_paras['meta_rl'] = train_paras['meta_rl']
    else:
        train_paras['meta_rl']['enable'] = False
    if opts.subprob:
        train_paras['subprob'] = True
    else:
        train_paras['subprob'] = False
    if opts.new_job:
        env_paras['new_job'] = True
        
    # =====================================================


    # ===== FJSP with Transformer =====
    if model_paras["algorithm"] == "hgs":  # this is a final proposed method
        trainer = TFJSPTrainer_GTrans(env_paras, model_paras, train_paras, optimizer_paras, test_paras, change_paras, model_version=2)
    elif model_paras["algorithm"] == "matnet":
        trainer = TFJSPTrainer_matnet(env_paras, model_paras, train_paras, optimizer_paras, test_paras, change_paras, model_version=1)
    elif model_paras["algorithm"] == "hgs_selfatten":
        trainer = TFJSPTrainer_DTrans(env_paras, model_paras, train_paras, optimizer_paras, test_paras, change_paras, model_version=10)
    elif model_paras["algorithm"] == "hgs_nograph":
        trainer = TFJSPTrainer_DTrans(env_paras, model_paras, train_paras, optimizer_paras, test_paras, change_paras, model_version=16)
    elif model_paras["algorithm"] == "hgnn":
        trainer = TFJSPTrainer_hgnn(env_paras, model_paras, train_paras, optimizer_paras, test_paras, change_paras)
    else:
        raise Exception("algorithm ", model_paras["algorithm"], " does not exist!!")
    trainer.run()
        
    
    
    
def _print_config(load_dict):
    logger = logging.getLogger('root')
    for key, val in load_dict.items():
        logger.info(key+ ": {}".format(val) + "\n")
    

if __name__ == '__main__':
    # A = torch.arange(2*3).reshape(2,3)
    # B = torch.arange(2*2*3).reshape(2,2,3)
    # C = A[:, None, :].expand(-1, 2, -1) + B
    # print(f'A:{A}')
    # print(f"B:{B}")
    # print(f"C:{C}")
    
    main(get_options())
    
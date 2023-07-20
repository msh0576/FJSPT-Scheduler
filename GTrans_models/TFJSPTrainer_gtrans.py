from json import encoder
import torch
import numpy as np
import os
from copy import deepcopy
from collections import deque
import random
import time
import pandas as pd
import math


from env.case_generator_v2 import CaseGenerator
from env.tfjsp_env import TFJSPEnv

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from logging import getLogger
from utils.utils import *

# from DHJS_models.TFJSPTrainer_dhjs import TFJSPTrainer_DHJS
# from DTrans_models.TFJSPModel_dtrans import TFJSPModel_DTrans
from DTrans_models.TFJSPTrainer_dtrans import TFJSPTrainer_DTrans
from GTrans_models.TFJSPModel_gtrans import TFJSPModel_GTrans

class TFJSPTrainer_GTrans(TFJSPTrainer_DTrans):
    def __init__(self,
                env_paras,
                model_paras,
                train_paras,
                optimizer_paras,
                test_paras,
                change_paras,
                model_version=1,
                ):
        super().__init__(
            env_paras,
            model_paras,
            train_paras,
            optimizer_paras,
            test_paras,
            change_paras,
        )
        self.env_paras = env_paras
        self.model_paras = model_paras
        self.train_paras = train_paras
        self.optimizer_paras = optimizer_paras
        self.test_paras = test_paras
        self.change_paras = change_paras
        
        # ===== encoder/decoder version =====
        if model_version == 1:
            encoder_version = 1
            decoder_version = 5
        elif model_version == 2:
            encoder_version = 2
            decoder_version = 5
        else:
            raise Exception('encoder/decoder version error!')
        # === Model ===
        self.model = TFJSPModel_GTrans(
            embedding_dim_=model_paras["embedding_dim"],
            hidden_dim_=model_paras["hidden_dim"],
            problem=None,
            ope_feat_dim=model_paras["in_size_ope"],
            ma_feat_dim=model_paras["in_size_ma"],
            veh_feat_dim=model_paras["in_size_veh"],
            mask_inner=True,
            mask_logits=True,
            encoder_version=encoder_version,
            decoder_version=decoder_version,
            meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
            **model_paras
        ).to(model_paras["device"])
        self.base_model = deepcopy(self.model)
        

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_paras['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_paras['scheduler'])        
        
        # === for debug model parameter change ===
        self.prev_model_para = None
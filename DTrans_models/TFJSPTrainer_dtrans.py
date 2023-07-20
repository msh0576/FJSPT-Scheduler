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

from DHJS_models.TFJSPTrainer_dhjs import TFJSPTrainer_DHJS
from DTrans_models.TFJSPModel_dtrans import TFJSPModel_DTrans


class TFJSPTrainer_DTrans(TFJSPTrainer_DHJS):
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
            decoder_version = 1
        elif model_version == 2:
            encoder_version = 2
            decoder_version = 1
        elif model_version == 3:
            encoder_version = 3
            decoder_version = 1
        elif model_version == 4:
            encoder_version = 3
            decoder_version = 4
        elif model_version == 5:
            encoder_version = 3
            decoder_version = 3
        elif model_version == 6:
            encoder_version = 4
            decoder_version = 4
        elif model_version == 7:
            encoder_version = 5
            decoder_version = 5
        elif model_version == 8:
            encoder_version = 0
            decoder_version = 5
        elif model_version == 9:
            encoder_version = 5
            decoder_version = 0
        elif model_version == 10:
            encoder_version = 1
            decoder_version = 5
        elif model_version == 11:
            encoder_version = 3
            decoder_version = 5
        elif model_version == 12:
            encoder_version = 6
            decoder_version = 5
        elif model_version == 13:
            encoder_version = 4
            decoder_version = 5
        elif model_version == 14:
            encoder_version = 7
            decoder_version = 5
        elif model_version == 15:
            encoder_version = 8
            decoder_version = 5
        elif model_version == 16:
            encoder_version = 0
            decoder_version = 0
        else:
            raise Exception('encoder/decoder version error!')

        self.encoder_version = encoder_version
        # ===== model, optimizer, scheduler ===
        self.model = TFJSPModel_DTrans(
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
        
    
        
    
    def _train_one_batch(self, batch_size, env, train_dataset=None, train_loader=None):
        # ===== Preparation =====
        self.model.train()
        state = env.reset()
        self.model.init(state)
        
        base_env = deepcopy(env)
        self.base_model.load_state_dict(self.model.state_dict())
        
        # ===== Rollout =====
        done = False
        dones = env.done_batch
        epi_log_p = torch.zeros(size=(batch_size, 0))
        all_rewards = torch.zeros(size=(batch_size,))
        while not done:
            action, log_p = self.model.act(state)   # [3, B] | [B, 1]
            # print(f'action:{action}')
            
            # action = random_act(state)
            # state, rewards, dones = env.step(action, step_reward='version2')
            state, rewards, dones = env.step(action)
            done = dones.all()
            epi_log_p = torch.cat([epi_log_p, log_p], dim=1)    # [B, T]
            all_rewards += rewards
        
        # ===== Learning =====
        baseline_value = self._baseline(base_env, self.base_model)
        advantage = rewards - baseline_value
        # advantage = rewards - rewards.mean()
        epi_log_p = epi_log_p.sum(dim=1)    # [B, 1]
        loss = -advantage * epi_log_p
        loss_mean = loss.mean()

        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        
        # === score ===
        # score = -rewards.mean()
        score = env.get_makespan().mean()
        
        
        return score.item(), loss_mean.item()
    
    def _baseline(self, env, model):
        model.eval()
        state = env.state
        model.init(state)
        
        done = False
        dones = env.done_batch
        while not done:
            with torch.no_grad():
                action, _ = model.act(state, baseline=True)   # [3, B] | [B, 1]
            state, rewards, dones = env.step(action)
            done = dones.all()
        return rewards
    
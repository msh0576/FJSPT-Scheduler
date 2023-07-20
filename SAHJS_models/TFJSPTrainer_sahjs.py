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

from DHJS_models.meta_learner import subgraph_meta_learner
from DHJS_models.TFJSPTrainer_dhjs import TFJSPTrainer_DHJS
from SAHJS_models.TFJSPModel_sahjs import TFJSPModel_SAHJS


from torch_geometric.utils import scatter

class TFJSPTrainer_SAHJS(TFJSPTrainer_DHJS):
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
            model_version
        )
        self.device = model_paras["device"]
        # === setup conf num_edge_feat ===
        self.proctime_per_ope_mas = env_paras["proctime_per_ope_max"]
        self.transtime_btw_ma_max = env_paras["transtime_btw_ma_max"]
        num_edge_feat = max(self.proctime_per_ope_mas, self.transtime_btw_ma_max) + 1
        model_paras['num_edge_feat'] = num_edge_feat
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
        else:
            raise Exception('encoder/decoder version error!')

        self.encoder_version = encoder_version
        # ===== model, optimizer, scheduler ===
        self.model = TFJSPModel_SAHJS(
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
        
        
        
        
    def _train_one_batch(self, batch_size, env,
                        train_dataset=None, train_loader=None):
        
        # ===== Preparation =====
        self.model.train()
        state = env.reset()
        self.model.init(state, train_dataset, train_loader)
        
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
        baseline_value = self._baseline(base_env, self.base_model, train_dataset, train_loader)
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
    
    def _baseline(self, env, model, train_dataset, train_loader):
        model.eval()
        state = env.state
        model.init(state, train_dataset, train_loader)
        
        done = False
        dones = env.done_batch
        while not done:
            with torch.no_grad():
                action, _ = model.act(state, baseline=True)   # [3, B] | [B, 1]
            state, rewards, dones = env.step(action)
            done = dones.all()
        return rewards
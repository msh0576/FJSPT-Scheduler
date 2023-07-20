import torch
import numpy as np
import os
import copy
from collections import deque
import random
import time
import pandas as pd
import math

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from logging import getLogger
from utils.utils import *

# from env.case_generator import CaseGenerator
from env.case_generator_v2 import CaseGenerator
from env.tfjsp_env import TFJSPEnv

from matnet_models.TFJSPModel_matnet import TFJSPModel_matnet
from matnet_models.TFJSPModel_matnet_jobnode import TFJSPModel_matnet_jobnode
from DHJS_models.TFJSPTrainer_dhjs import TFJSPTrainer_DHJS


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class TFJSPTrainer_matnet(TFJSPTrainer_DHJS):
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
        self.model = TFJSPModel_matnet(
            embedding_dim_=self.model_paras["embedding_dim"],
            ope_feat_dim=self.model_paras["in_size_ope"],
            ma_feat_dim=self.model_paras["in_size_ma"],
            veh_feat_dim=self.model_paras["in_size_veh"],
            mask_inner=True,
            mask_logits=True,
            **model_paras
        ).to(self.device)
        
    
    def _train_one_batch(self, batch_size, env, 
                        train_dataset=None, train_loader=None):
        # ===== Preparation =====
        # self.model.train()
        state = env.reset()
        self.model.init(state)
        
        # base_env = copy.deepcopy(env)
        # base_model = copy.deepcopy(self.model)
        # base_model.load_state(self.model.state_dict())
        
        prob_list = torch.zeros(size=(batch_size, 0))
        
        # ===== Rollout =====
        done = False
        dones = env.done_batch
        while not done:
            action, prob = self.model.act(state)   # [3, B] | [B, 1]
            # action = self.model.random_act(state)
            state, rewards, dones = env.step(action)
            done = dones.all()
            prob_list = torch.cat((prob_list, prob), dim=1)
        # === Verify the solution ===
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            self.logger.info("Scheduling Error!!!!!")
        
        # env.render(spec_batch_id=0)
        
        # === baseline ===
        # baseline_value = self._baseline(base_env, base_model)

        # ===== Learning =====
        advantage = rewards - rewards.mean()
        # advantage = rewards - baseline_value
        log_prob = prob_list.log().sum(dim=1)
        loss = -advantage * log_prob
        loss_mean = loss.mean()
        # print(f'loss_mean:{loss_mean:0.1f}')

        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        
        # === score ===
        score = env.get_makespan().mean()
        
        
        return score.item(), loss_mean.item()
    
    def _baseline(self, env, model):
        state = env.state
        done = False
        dones = env.done_batch
        while not done:
            with torch.no_grad():
                action, _ = model(state, baseline=True)   # [3, B] | [B, 1]
            state, rewards, dones = env.step(action)
            done = dones.all()
        return rewards
    
    
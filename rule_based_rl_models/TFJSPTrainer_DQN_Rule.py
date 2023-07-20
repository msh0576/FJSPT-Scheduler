import torch
import torch.nn as nn
import numpy as np
import os
from copy import deepcopy
from collections import deque
import random
import time
import pandas as pd
import math

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from matnet_v2_models.TFJSPTrainer_matnet_v2 import TFJSPTrainer_matnet_v2
from rule_based_rl_models.TFJSPModel_DQN_Rule import TFJSPModel_DQN_Rule
from utils.memory import ReplayBuffer

class TFJSPTrainer_DQN_Rule(TFJSPTrainer_matnet_v2):
    def __init__(self,
                 env_paras,
                model_paras,
                train_paras,
                optimizer_paras,
                test_paras,
                change_paras,
                 ):
        super().__init__(
            env_paras,
            model_paras,
            train_paras,
            optimizer_paras,
            test_paras,
            change_paras,
        )
        
        self.device = model_paras['device']

        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras['gamma']
        self.K_epochs = train_paras['K_epochs']
        self.eps_clip = train_paras['eps_clip'] # clip ratio for PPO
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        
        
        model_paras["actor_in_dim"] = env_paras['num_jobs'] + env_paras['num_mas'] + env_paras['num_vehs']
        model_paras["critic_in_dim"] = env_paras['num_jobs'] + env_paras['num_mas'] + env_paras['num_vehs']
        
        # self.model = DQN_Rule(
        #     model_paras,
        #     train_paras
        # )
        self.model = TFJSPModel_DQN_Rule(model_paras, train_paras).to(self.device)
        self.model_old = deepcopy(self.model)
        self.model_old.load_state_dict(self.model.state_dict())
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_paras['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_paras['scheduler'])   
        
        # === replay memory ===
        self.memory = ReplayBuffer(self.device, max_size=int(1e5))
        
    
    def _train_one_batch(self, batch_size, env):
        # ===== Preparation =====
        self.model.train()
        self.model_old.train()
        state = env.reset()
        self.model.init(state)
        self.model_old.init(state)
        
        # ===== Rollout =====
        done = False
        dones = env.done_batch
        epi_log_p = torch.zeros(size=(batch_size, 0))
        while not done:
            # : memorize state
            self.memory.add_state_info(state)
            # : get action
            action, log_p = self.model_old.act(state, memory=self.memory)   # [4, B] | [B, 1]
            # : step enviornment
            state, rewards, dones = env.step(action, step_reward=True)
            done = dones.all()
            epi_log_p = torch.cat([epi_log_p, log_p], dim=1)    # [B, T]
            # : memorize reward
            self.memory.add_reward_info(rewards.detach().cpu().numpy(), dones.detach().cpu().numpy(), batch_size)

        # === learning ===
        loss = self.model.update(self.memory, self.optimizer, self.train_paras['minibatch_size'],
                          self.gamma, self.K_epochs, self.eps_clip, 
                          self.A_coeff, self.vf_coeff, self.entropy_coeff)
        # Copy new weights into old policy
        self.model_old.load_state_dict(self.model.state_dict())
        self.memory.clear()
        
        # === score ===
        score = env.get_makespan().mean()
        
        # === debug ===
        # : init_embed_opes.weight
        # : actor.linears.0.weight
        # : critic.linears.0.weight
        # for name, param in self.model.policy.named_parameters():
        #     # print(name)
        #     if name == "actor.linears.0.weight" and self.prev_model_para is None:
        #         print(name, "\n", param[0])
        #         self.prev_model_para = copy.deepcopy(param)
        #     elif name == "actor.linears.0.weight" and not torch.equal(param, self.prev_model_para):
        #         print(name, "\n", param[0])
        #         self.prev_model_para = copy.deepcopy(param)

        return score.item(), loss
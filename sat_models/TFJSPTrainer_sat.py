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


from torch_geometric.loader.dataloader import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils

from DHJS_models.TFJSPTrainer_dhjs import TFJSPTrainer_DHJS
from sat_models.TFJSPModel_sat import GraphTransformer
from sat_models.sat.gnn_layers import GNN_TYPES
from sat_models.graph_dataset import set_GraphData
from sat_models.sat.data import GraphDataset

class TFJSPTrainer_sat(TFJSPTrainer_DHJS):
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
        batch_size = env_paras['batch_size']
        device = env_paras['device']
        
        # === get env ===
        env = self._generate_env(
            self.num_opes, self.num_mas, self.num_vehs, self.device, self.dynamic,
            self.proctime_per_ope_mas, self.transtime_btw_ma_max
        )
        # === get graph data: for all batch ===
        graph_dataset = []
        for batch in range(batch_size):
            data = set_GraphData(
                self.num_opes, self.num_mas, self.num_vehs, env.nums_ope_batch[batch],
                env.ope_ma_adj_batch[batch], env.proc_times_batch[batch], env.trans_times_batch[batch],
                node_feat_dim=8, edge_feat_dim=1
            )
            graph_dataset.append(data)

        self.train_dataset = GraphDataset(graph_dataset, degree=True, 
            k_hop=1, se='khopgnn',
            use_subgraph_edge_attr=True
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True
        )
    
        # === get degree ===
        deg = torch.cat([
            utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for
            data in self.train_dataset]
        )
        # === define model ===
        n_nodes = env_paras['num_opes'] + env_paras['num_mas'] +env_paras['num_vehs']
        model_paras['num_heads'] = 8
        self.model = GraphTransformer(
            in_size=n_nodes,
            num_class=1,
            d_model=self.model_paras["embedding_dim"],
            dim_feedforward=2*self.model_paras["embedding_dim"],
            # dropout=0.2,
            # num_heads=8,
            num_layers=6,
            batch_norm=True,
            # abs_pe=None,
            # abs_pe_dim=20,
            gnn_type='pna2',
            use_edge_attr=True,
            num_edge_features=21,   # edge 최대 값
            # edge_dim=32,
            k_hop=1,
            se='khopgnn',
            deg=deg,
            global_pool='mean',
            # device=device,
            ope_feat_dim=env_paras['ope_feat_dim'],
            ma_feat_dim=env_paras['ma_feat_dim'],
            veh_feat_dim=env_paras['veh_feat_dim'],
            encoder_version=10,
            decoder_version=1,
            # batch_size=env_paras['batch_size'],
            **model_paras
        )
        
        
        self.base_model = deepcopy(self.model)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_paras['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_paras['scheduler'])    
        
    def _train_one_batch(self, batch_size, env,
                        train_dataset=None, train_loader=None):
        # ===== Preparation =====
        self.model.train()
        state = env.reset()
        self.model.init(state, self.train_dataset)
        
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
        model.init(state, self.train_dataset)
        
        done = False
        dones = env.done_batch
        while not done:
            with torch.no_grad():
                action, _ = model.act(state, baseline=True)   # [3, B] | [B, 1]
            state, rewards, dones = env.step(action)
            done = dones.all()
        return rewards
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple
import math
import numpy as np

from TFJSPModel_dhjs import TFJSPModel_DHJS

class TFJSPModel_subprob(TFJSPModel_DHJS):
    def __init__(self,
                embedding_dim_,
                hidden_dim_,
                problem,
                ope_feat_dim,
                ma_feat_dim,
                veh_feat_dim,
                n_encode_layers=2,
                tanh_clipping=10.,
                mask_inner=True,
                mask_logits=True,
                normalization='batch',
                n_heads=8,
                checkpoint_encoder=False,
                shrink_size=None,
                consd_trans_time_mat=True,
                encoder_version=1,
                decoder_version=1,
                meta_rl=None,
                **model_paras
                ):
        '''
        Input:
            meta_rl: train_paras['meta_rl']
        '''
        super().__init__(
            embedding_dim_,
            hidden_dim_,
            problem,
            ope_feat_dim,
            ma_feat_dim,
            veh_feat_dim,
            n_encode_layers=2,
            tanh_clipping=10.,
            mask_inner=True,
            mask_logits=True,
            normalization='batch',
            n_heads=8,
            checkpoint_encoder=False,
            shrink_size=None,
            consd_trans_time_mat=True,
            encoder_version=1,
            decoder_version=1,
            meta_rl=None,
            **model_paras
        )
    
    def init(self, state, dataset=None):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)
        nums_ope_batch = state.nums_ope_batch
        proc_times_batch = state.proc_times_batch
        ope_ma_adj_batch = state.ope_ma_adj_batch
        trans_times_batch = state.trans_times_batch
        
        # === subproblem trining ===
        if self.encoder_version == 11:
            # generate subproblems
            
            pass
        else:
            raise Exception('encoder_version error')
    
    def forward(self, state, baseline=False):
        batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_jobs = state.mask_job_finish_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_vehs = state.mask_veh_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]

        # === embedding ===
        # if encoder_version==3, outputs embedding list: [T, B, N, D_emb]
        # batch_core_adj_list: batch list: [max_k_core, n_nodes, n_nodes]
        embed_feat_ope, embed_feat_ma,\
            embed_feat_veh, norm_proc_trans_time, norm_offload_trans_time, \
            norm_trans_time, oper_adj_batch, _, norm_MVpair_trans_time, norm_onload_trans_time, \
            mask_dyn_ope_ma_adj, mask_ma \
            = self.embedder.embedding(state, self.encoder_version)
        
        # === encoding ===
        embedded_ope, embedded_ma, embedded_veh = self.encoder(
            embed_feat_ope, embed_feat_ma, embed_feat_veh, 
            norm_proc_trans_time, norm_offload_trans_time, norm_trans_time, oper_adj_batch,
            self.batch_core_adj_list, norm_MVpair_trans_time, norm_onload_trans_time,
            mask_dyn_ope_ma_adj, mask_ma
        )    # [B, n_opes, D_emb] | [B, n_mas, D_emb]
        
        # === decoding ===
        action, log_p = self._get_action_with_decoder(state, embedded_ope, embedded_ma, embedded_veh, baseline=baseline)
        # print(f'action:{action} | log_p:{log_p}')
        
        # action = random_act(state)
        return action, log_p
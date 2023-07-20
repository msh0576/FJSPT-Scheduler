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
import torch_geometric.utils as utils

from env.common_func import generate_trans_mat, random_act
from SAHJS_models.embedder import SAHJS_embedder
from SAHJS_models.encoder import TFJSP_Encoder_SAHJS, StructureAwareEncoderLayer
from DHJS_models.decoder_types import TFJSP_Decoder_DHJS_V2, TFJSP_Decoder_DHJS_V3
from DHJS_models.decoder import TFJSP_Decoder_DHJS_Base
from DHJS_models.utils import get_core_adj_list
from DHJS_models.subgraphs import get_core_adj_mat
from DHJS_models.TFJSPModel_dhjs import TFJSPModel_DHJS
from sat_models.graph_dataset import new_edge_attr, get_edge_attr
from sat_models.sat.position_encoding import POSENCODINGS
from SAHJS_models.position_encoding import LapEncoding
from SAHJS_models.temporal_graph_dataset import get_EdgeAttr_from_EdgeIndex, \
    comp_edge_index_bias, get_EdgeAttr_from_EdgeIndexBatch, make_opes_list, build_ope_edge_tensor



class TFJSPModel_SAHJS(TFJSPModel_DHJS):
    '''
    This version improve training speed, where only selected job node, not operation nodes, computes nearest vehicle nodes
    
    '''
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
        super().__init__(
            embedding_dim_,
            hidden_dim_,
            problem,
            ope_feat_dim,
            ma_feat_dim,
            veh_feat_dim,
            n_encode_layers,
            tanh_clipping,
            mask_inner,
            mask_logits,
            normalization,
            n_heads,
            checkpoint_encoder,
            shrink_size,
            consd_trans_time_mat,
            encoder_version,
            decoder_version,
            meta_rl,
            **model_paras
        )
        
        batch_size = model_paras['batch_size']
        abs_pe_method = POSENCODINGS['rw']
        self.abs_pe = model_paras['abs_pe']
        abs_pe_dim = model_paras['abs_pe_dim']
        # === embedder ===
        self.embedder = SAHJS_embedder(
            embedding_dim_, self.ope_feat_dim, self.ma_feat_dim, self.veh_feat_dim,
            **model_paras
        )
        # === edge embedder ===
        # edge_dim = model_paras['edge_dim']
        num_edge_feat = model_paras['num_edge_feat']
        self.embedding_edge = nn.Embedding(num_edge_feat, embedding_dim_)
        # === encoder ===
        self.encoder_version = encoder_version
        
        if encoder_version in [1]:
            encoder_fn = TFJSP_Encoder_SAHJS
        elif encoder_version in [2, 3]:
            encoder_fn = StructureAwareEncoderLayer
        else:
            raise Exception('encoder_version error!')
        self.encoder = encoder_fn(
            encoder_version, **model_paras
        )
        # === decoder ===
        self.decoder_version = decoder_version
        self.prev_embed = torch.zeros(size=(batch_size, 1, self.embedding_dim))
        if decoder_version == 1:
            decoder_fn = TFJSP_Decoder_DHJS_Base
            self.prev_embed = torch.zeros(size=(batch_size, 1, 3*self.embedding_dim))
        elif decoder_version == 2:
            decoder_fn = TFJSP_Decoder_DHJS_V2
        elif decoder_version == 3:
            decoder_fn = TFJSP_Decoder_DHJS_V3
            self.prev_embed = torch.zeros(size=(batch_size, 1, 3*self.embedding_dim))
        else:
            raise Exception('decoder version error!')
        self.decoder = decoder_fn(**model_paras)
        
        # === position encoder ===
        self.abs_pe_encoder = LapEncoding(abs_pe_dim, use_edge_attr=True, normalization='rw')
    
    def init(self, state, dataset=None, loader=None):
        '''
        :param dataset: DataBatch()
        '''
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)
        self.nums_ope = state.nums_ope_batch
        num_nodes = self.num_opes + self.num_mas + self.num_vehs
            
        for batch_idx, batch_data in enumerate(loader):
            self.edge_index_batch = batch_data.batch[batch_data.edge_index[1]]   # [total_n_edge]: element values = 해당 edge_index의 batch_index
        self.dataset = batch_data
        self.fixed_edge_index = []
        self.fixed_subgraph_node_index = []
        self.fixed_subgraph_edge_index = []
        self.fixed_subgraph_indicator_index = []
        self.fixed_pos_enc = []
        
        # === fixed ope_ope_adj ===
        self.ope_adj_mat = state.ope_pre_adj_batch.long()   # [B, n_opes, n_opes]
        self.edge_mat = torch.zeros(size=(self.batch_size, num_nodes, num_nodes))
        self.edge_mat[:, :self.num_opes, :self.num_opes] = self.ope_adj_mat
        
        # === fixed position encoding ===
        total_n_nodes = self.dataset.x.size(0)
        self.lap_pe = None
        if self.abs_pe:
            self.lap_pe = self.abs_pe_encoder.compute_pe(
                self.dataset.edge_index.cpu(), self.dataset.edge_attr.cpu(), 
                total_n_nodes).to(self.device)
        
    
    def act(self, state, baseline=False):
        return self.forward(state, baseline)
    
    def forward(self, state, baseline=False):
        batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_jobs = state.mask_job_finish_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_vehs = state.mask_veh_procing_batch.size(1)
        num_nodes = num_opes + num_mas + num_vehs
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]

        # === embedding ===
        embed_feat_node, proc_time, trans_time, offload_trans_time \
            = self.embedder.embedding(state, self.encoder_version)
        offload_trans_time = offload_trans_time+1
        embed_feat_node = embed_feat_node.reshape(batch_size*num_nodes, -1) 
        
        # === temporal edge_attr ===      
        self.edge_mat[:, :num_opes, num_opes:num_opes+num_mas] = proc_time  # [B, n_opes, n_mas]
        self.edge_mat[:, num_opes:num_opes+num_mas, num_opes+num_mas:] = offload_trans_time # [B, n_mas, n_vehs]
        self.edge_mat[:, num_opes:num_opes+num_mas, num_opes:num_opes+num_mas] = trans_time # [B, n_mas, n_mas]
        symm_edge_mat = self.edge_mat + self.edge_mat.transpose(1,2)
        
        edge_index, edge_attr = utils.dense_to_sparse(symm_edge_mat)
        matching_indices = torch.where((self.dataset.edge_index[0].unsqueeze(1) == edge_index[0]) &
                               (self.dataset.edge_index[1].unsqueeze(1) == edge_index[1]))[0]
        non_matching_indices = torch.arange(self.dataset.edge_index.size(1))
        non_matching_indices = non_matching_indices[~torch.isin(non_matching_indices, matching_indices)]

        self.dataset.edge_attr[matching_indices] = edge_attr.long()
        self.dataset.edge_attr[non_matching_indices] = 0
        
        # edge_attr = self.embedding_edge(self.dataset.edge_attr).long()  # [n_edge, D_emb_edge]
        # subgraph_edge_attr = self.embedding_edge(self.dataset[i].subgraph_edge_attr).long()
        # pos_enc = self.embedding_abs_pe(self.dataset.abs_pe)    # [n_node, D_emb]
        
        # === position encoding ===
        # self.abs_pe_encoder
        
        # === encoding ===
        embedded_node = self.encoder(
            x=embed_feat_node,
            edge_index=self.dataset.edge_index,
            edge_attr=self.dataset.edge_attr,
            batch=self.dataset.batch,
            num_opes=num_opes, num_mas=num_mas, num_vehs=num_vehs,
            batch_size=batch_size,
            pe=self.lap_pe,
        )
        embedded_node = embedded_node.reshape(batch_size, num_nodes, -1)
        
        embedded_ope = embedded_node[:, :num_opes]
        embedded_ma = embedded_node[:, num_opes:num_opes+num_mas]
        embedded_veh = embedded_node[:, num_opes+num_mas:]
        
        
        # === decoding ===
        action, log_p = self._get_action_with_decoder(state, embedded_ope, embedded_ma, embedded_veh, baseline=baseline)
        
        return action, log_p
    
    def _get_target_edge_indicator(self):
        '''
        :return ope_ma_edge_index: [3, n_ope_ma_edge_batch]: batch에 따라 bias 된 edge_index
        :return ope_ma_edge_indicator:  전체 edge_index [2, n_edge_batch] 중 ope_ma_edge에 해당하는 column indicator
        :return ma_veh_edge_index:
        :return ma_veh_edge_indicator:
        '''
        num_nodes_batch = self.dataset.edge_index_bias['num_nodes']   # [n_batch,]
        edge_index_bias = comp_edge_index_bias(num_nodes_batch.tolist()) # [n_batch,]
        edge_index_bias = torch.tensor(edge_index_bias).long()
        tmp_edge_index = self.dataset.edge_index - edge_index_bias[self.edge_index_batch]    # [2, n_edge_batch]
        # [3, n_edge_batch]: batch 들의 each_index 를 기존 node_id로 bias 시킴
        augdim_edge_index = torch.cat([self.edge_index_batch[None, :], tmp_edge_index], dim=0)  
        
        edge_pos_bias = self.dataset.edge_index_bias['num_edges']   # [n_batch,]
        edge_pos_bias = comp_edge_index_bias(edge_pos_bias.tolist())
        edge_pos_bias = torch.tensor(edge_pos_bias).long()
        ope_ma_edge_bias = self.dataset.edge_index_bias['ope_ma_edge_bias'] # [n_batch,]
        ma_veh_edge_bias = self.dataset.edge_index_bias['ma_veh_edge_bias']
        ma_ma_edge_bias = self.dataset.edge_index_bias['ma_ma_edge_bias']
        
        ope_ma_edge_bias = edge_pos_bias + ope_ma_edge_bias # [n_batch,]
        ma_veh_edge_bias = edge_pos_bias + ma_veh_edge_bias
        ma_ma_edge_bias = edge_pos_bias + ma_ma_edge_bias
        
        ope_ma_edge_indicator = [torch.arange(start, end) for start, end in zip(ope_ma_edge_bias, ma_veh_edge_bias)]
        ope_ma_edge_indicator = torch.cat(ope_ma_edge_indicator)    # [n_ope_ma_edge_batch,]
        ope_ma_edge_index = augdim_edge_index[:, ope_ma_edge_indicator] # [3, n_ope_ma_edge_batch]
        
        ma_veh_edge_indicator = [torch.arange(start, end) for start, end in zip(ma_veh_edge_bias, ma_ma_edge_bias)]
        ma_veh_edge_indicator = torch.cat(ma_veh_edge_indicator)    # [n_ope_ma_edge_batch,]
        ma_veh_edge_index = augdim_edge_index[:, ma_veh_edge_indicator] # [3, n_ope_ma_edge_batch]
        
        return ope_ma_edge_index, ope_ma_edge_indicator, \
            ma_veh_edge_index, ma_veh_edge_indicator
            
    def _get_ope_adj_mat(self, nums_ope):
        '''
        :param nums_ope: [B, n_jobs]
        :return ope_adj_mat: [B, n_opes, n_opes]
        '''
        batch_size = nums_ope.size(0)
        
        for i in range(batch_size):
            opes_list = make_opes_list(nums_ope[i])
            ope_edge_idx, ope_edge_weig = build_ope_edge_tensor(opes_list) 
        pass
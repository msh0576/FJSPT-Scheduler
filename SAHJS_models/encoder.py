

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from env.common_func import reshape_by_heads
from torch_geometric.utils import to_dense_adj

from DHJS_models.encoding_block import EncodingBlock_Base, EncodingBlock_Job, SelfEncodingBlock
from matnet_models.FFSPModel_SUB import AddAndInstanceNormalization, FeedForward, MixedScore_MultiHeadAttention
from SAHJS_models.encoder_block import TransformerEncoder
from SAHJS_models.position_encoding import LapEncoding

class TFJSP_Encoder_SAHJS(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        
        embedding_dim = model_params['embedding_dim']
        num_layers = model_params['encoder_layer_num']
        abs_pe_dim = model_params['abs_pe_dim']
        num_edge_feat = model_params['num_edge_feat']
        
        self.pos_enc = nn.Linear(abs_pe_dim, embedding_dim)
        self.embedding_edge = nn.Embedding(num_edge_feat, embedding_dim)
        
        if encoder_version == 1:
            self.encoding_block = TransformerEncoder(embedding_dim, num_layers, **model_params)
        else:
            raise Exception('encoder_version error!')

    def forward(self, 
            x, edge_index, edge_attr, 
            num_opes, num_mas, num_vehs, batch_size,
            batch=None, pe=None,
        ):
        '''
        :param x: [n_node, D_emb]
        :param edge_index: [2, n_edge]
        :param edge_attr: [n_edge, D_emb_edge]
        :param subgraph_node_index: 
        :param pos_enc: []
        
        '''
        # pe = self.pos_enc(pos_enc)
        edge_attr = self.embedding_edge(edge_attr).long()  # [n_edge, D_emb_edge]
        
        x = self.encoding_block(
            x=x,
            pe=pe,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )
        
        return x


class StructureAwareEncoderLayer(nn.Module):
    '''
    encoder version 2, 3
    '''
    def __init__(self, 
        encoder_version, **model_params
    ):
        super().__init__()
        num_layers = model_params['encoder_layer_num']
        
        self.layers = nn.ModuleList([StructureAwareEncoder(encoder_version, **model_params) for _ in range(num_layers)])
        
    def forward(
        self, x, edge_index, edge_attr, 
        num_opes, num_mas, num_vehs, batch_size,
        batch=None, pe=None,
    ):
        
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, 
                num_opes, num_mas, num_vehs, batch_size,
                batch, pe
            )
        return x
        
class StructureAwareEncoder(nn.Module):
    '''
    encoder version 2, 3
    '''
    def __init__(self, 
        encoder_version, **model_params
    ):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        self.device= model_params['device']
        
        embedding_dim = model_params['embedding_dim']
        num_layers = model_params['encoder_layer_num']
        abs_pe_dim = model_params['abs_pe_dim']
        num_edge_feat = model_params['num_edge_feat']

        # self.abs_pe_encoder = LapEncoding(abs_pe_dim, use_edge_attr=True, normalization='rw')
        self.pos_enc = nn.Linear(abs_pe_dim, embedding_dim)
        self.embedding_edge = nn.Embedding(num_edge_feat, embedding_dim)

        self.ma_encoding_block = EncodingBlock_Base(**model_params)
        self.veh_encoding_block = EncodingBlock_Base(**model_params)
        self.job_encoding_block = EncodingBlock_Base(**model_params)
        
        if encoder_version == 3:
            self.global_encoding_block = SelfEncodingBlock(**model_params)
    
    def forward(
        self, x, edge_index, edge_attr, 
        num_opes, num_mas, num_vehs, batch_size,
        batch=None, pe=None,
    ):
        '''
        :param x: [n_node, -1]
        :param edge_index: [2, n_edge]
        :param edge_attr: [n_edge,]
        
        
        '''
        n_nodes = num_opes + num_mas + num_vehs
        
        # === position encoding ===
        if pe is not None:
            x = x + self.pos_enc(pe)
        # === get adj_mat ===
        x = x.reshape(batch_size, n_nodes, -1)
        ope_emb = x[:, :num_opes, :]
        ma_emb = x[:, num_opes:num_opes+num_mas, :]
        veh_emb = x[:, num_opes+num_mas:, :]
        edge_attr = self.embedding_edge(edge_attr).long()  # [n_edge, D_emb_edge]
        
        adj_mat = to_dense_adj(
            edge_index, 
            batch=batch,
            edge_attr=edge_attr,
            # max_num_nodes=n_nodes,
        )   # [B, n_nodes, n_nodes, D_emb_edge]
        adj_mat = adj_mat.sum(dim=-1)   # [B, n_nodes, n_nodes]
        
        # === subgraph encoding ===
        ope_ma_veh_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1)    # [B, n_nodes, D_emb]
        tmp_ma_emb_out = self.ma_encoding_block(ma_emb, ope_ma_veh_emb, adj_mat[:, num_opes:num_opes+num_mas, :])   # [B, n_mas, E]
        tmp_veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, adj_mat[:, num_opes+num_mas:, num_opes:num_opes+num_mas])   # [B, n_vehs, E]
        tmp_ope_emb_out = self.job_encoding_block(ope_emb, ma_emb, adj_mat[:, :num_opes, num_opes:num_opes+num_mas])   # [B, n_jobs, E]
        x_out = torch.cat([tmp_ope_emb_out, tmp_ma_emb_out, tmp_veh_emb_out], dim=1).reshape(batch_size*n_nodes, -1)    
        
        
        # === global encoding ===
        if self.encoder_version == 3:
            x_out = x_out.reshape(batch_size, n_nodes, -1)
            x_out = self.global_encoding_block(x_out, x_out)
            x_out = x_out.reshape(batch_size*n_nodes, -1)
        
        return x_out
        
        


class EncoderLayer_Base(nn.Module):
    '''
    encoder version 1
    '''
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        
        embedding_dim = model_params['embedding_dim']
        num_layers = model_params['encoder_layer_num']
        
        if encoder_version == 1:
            self.encoding_block = TransformerEncoder(embedding_dim, num_layers, **model_params)
        else:
            raise Exception('encoder_version error!')
    
    def forward(self, 
            x, edge_index, edge_attr, 
            subgraph_node_index, subgraph_edge_index,
            subgraph_indicator_index, subgraph_edge_attr
        ):
        '''
        :param x: [n_node, D_emb]
        :param edge_index: [2, n_edge]
        :param edge_attr: [n_edge, D_emb_edge]
        :param subgraph_node_index: 
        
        '''
        
        
    
    def forward_tmp(self, 
            ope_emb, ma_emb, veh_emb, 
            proc_time, offload_trans_time, trans_time, 
            oper_adj_batch=None, batch_core_adj_mat=None,
            MVpair_trans=None, onload_trans_time=None,
            mask_dyn_ope_ma_adj=None, mask_ma=None,
        ):
        '''
        :param ope_emb: [B, n_opes, E]
        :param ma_emb: [B, n_mas, E]
        :param veh_emb: [B, n_vehs, E]
        :param proc_time_mat: [B, n_opes, n_mas]
        :param empty_trans_time_mat: [B, n_opes, n_vehs]
        :param trans_time_mat: [B, n_mas, n_mas]
        :param oper_adj_batch: [B, n_opes, n_opes]
        :param batch_core_adj_mat: [B, num_core, n_nodes, n_nodes]
        :param MVpair_trans_time [B, n_vehs, n_mas]
        :param onload_trans_time [B, n_opes, n_mas]
        :param mask_dyn_ope_ma_adj [B, n_opes, n_mas]
        :param mask_ma [B, n_mas]
        
        
        :return ope_emb_out: [B, n_opes, E]
        :return ma_emb_out: [B, n_mas, E]
        :return veh_emb: [B, n_vehs, E]
        '''
        num_opes = ope_emb.size(1)
        num_mas = ma_emb.size(1)
        num_vehs = veh_emb.size(1)
        
        proc_trans_time = torch.cat([proc_time.transpose(1, 2), trans_time], dim=-1)    # [B, n_mas, n_opes+n_mas]
        ope_ma_emb = torch.cat([ope_emb, ma_emb], dim=1)    # [B, n_opes+n_mas, D_emb]
        ma_emb_out = self.ma_encoding_block(ma_emb, ope_ma_emb, proc_trans_time)   # [B, n_mas, E]
        veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, MVpair_trans)   # [B, n_vehs, E]
        ope_emb_out = self.job_encoding_block(ope_emb, ma_emb, veh_emb, proc_time, offload_trans_time)   # [B, n_jobs, E]
        
        return ope_emb_out, ma_emb_out, veh_emb_out
        
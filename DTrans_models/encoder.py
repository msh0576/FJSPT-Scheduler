from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical
from copy import deepcopy
import torch_geometric.nn as gnn

from DHJS_models.encoding_block import EncodingBlock_Base, SelfEncodingBlock


class TFJSP_Encoder_DTrans(nn.Module):
    def __init__(self, 
                encoder_version, num_edge_feat_proc, 
                num_edge_feat_onload, num_edge_feat_offload,
                **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.encoder_version = encoder_version
        embedding_dim = model_params['embedding_dim']
        edge_dim = model_params['edge_dim']
        if encoder_version in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            self.layers = nn.ModuleList([EncoderLayer_Base(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        else:
            raise Exception('encoder version error!')
        
        # if encoder_version in [1, 2]:
        #     self.edge_embed_proc = nn.Embedding(num_edge_feat_proc, edge_dim)
        #     self.edge_embed_onload = nn.Embedding(num_edge_feat_onload, edge_dim)
        #     self.edge_embed_offload = nn.Embedding(num_edge_feat_offload, edge_dim)
        
        if encoder_version in [3, 5]:
            self.global_encoder = SelfEncodingBlock(**model_params)
        elif encoder_version in [7]:
            self.global_encoder = EncodingBlock_Base(**model_params)
        

    def forward(self, 
            ope_emb, ma_emb, veh_emb, 
            proc_time, offload_trans_time, onload_trans_time, 
            offload_trans_time_OV
        ):
        '''
        :param ope_emb: [B, n_opes, D_emb]
        :param ma_emb: [B, n_mas, D_emb]
        :param veh_emb: [B, n_vehs, D_emb]
        :param proc_time: [B, n_opes, n_mas]
        :param offload_trans_time: [B, n_vehs, n_mas]
        :param onload_trans_time: [B, n_mas, n_mas]
        :param offload_trans_time_OV: [B, n_vehs, n_jobs]
        '''
        n_opes = ope_emb.size(1)
        n_mas = ma_emb.size(1)
        n_vehs = veh_emb.size(1)
        # === edge embedding ===
        embed_proc_time = proc_time
        embed_offload_time = offload_trans_time
        embed_onload_time = onload_trans_time
        
        # === local encoding ===
        for layer in self.layers:
            ope_emb, ma_emb, veh_emb = layer(
                ope_emb, ma_emb, veh_emb, 
                embed_proc_time, embed_offload_time, embed_onload_time, 
                offload_trans_time_OV
            )

        # === global encoding ===
        if self.encoder_version in [3, 5, 7]:
            node_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1) # [B, n_nodes, D_emb]
            
            if self.encoder_version == 7:
                batch_size, n_nodes, _ = node_emb.size()
                global_edge = torch.zeros(size=(batch_size, n_nodes, n_nodes))
                global_edge[:, :n_opes, n_opes:n_opes+n_mas] = embed_proc_time
                global_edge[:, n_opes:n_opes+n_mas, :n_opes] = embed_proc_time.transpose(1,2)
                global_edge[:, n_opes:n_opes+n_mas, n_opes:n_opes+n_mas] = embed_onload_time
                global_edge[:, n_opes+n_mas:, :n_opes] = offload_trans_time_OV
                global_edge[:, :n_opes, n_opes+n_mas:] = offload_trans_time_OV.transpose(1,2)
                
                
                
                node_emb_out = self.global_encoder(node_emb, node_emb, global_edge)
            else:
                node_emb_out = self.global_encoder(node_emb, node_emb)
            ope_emb = node_emb_out[:, :n_opes, :]
            ma_emb = node_emb_out[:, n_opes:n_opes+n_mas, :]
            veh_emb = node_emb_out[:, n_opes+n_mas:, :]
        
        return ope_emb, ma_emb, veh_emb

class EncoderLayer_Base(nn.Module):
    '''
    encoder version 1, 2, 3, 4, 5, 6, 7
    '''
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        
        # === subgraph encoder ===
        if self.encoder_version in [2, 3, 4, 5, 6, 7, 8, 9]:
            self.ma_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
            self.ope_encoding_block = EncodingBlock_Base(**model_params)
        elif self.encoder_version == 1:
            self.node_encoding_block = SelfEncodingBlock(**model_params)
        if self.encoder_version == 6:
            self.global_encoding_block = SelfEncodingBlock(**model_params)
        elif self.encoder_version == 8:
            self.global_ma_encoding_block = SelfEncodingBlock(**model_params)
            self.global_veh_encoding_block = SelfEncodingBlock(**model_params)
            self.global_ope_encoding_block = SelfEncodingBlock(**model_params)
        elif self.encoder_version == 9:
            # self.global_encoding_block = nn.Linear(embedding_dim, embedding_dim)
            pass
        
    def forward(self, 
            ope_emb, ma_emb, veh_emb, 
            proc_time, offload_trans_time, onload_trans_time, 
            offload_trans_time_OV
        ):
        '''
        :param ope_emb: [B, n_opes, D_emb]
        :param ma_emb: [B, n_mas, D_emb]
        :param veh_emb: [B, n_vehs, D_emb]
        :param proc_time: [B, n_opes, n_mas]
        :param offload_trans_time: [B, n_vehs, n_mas]
        :param onload_trans_time: [B, n_mas, n_mas]
        :param offload_trans_time_OV: [B, n_vehs, n_jobs]
        
        '''
        n_opes = ope_emb.size(1)
        n_mas = ma_emb.size(1)
        n_vehs = veh_emb.size(1)
            
        if self.encoder_version in [2,3]:   # ope-ma-veh 구조
            ope_ma_veh_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1)    # [B, n_nodes, D_emb]
            proc_on_off_time = torch.cat(
                [proc_time.transpose(1,2), onload_trans_time, offload_trans_time.transpose(1,2)], 
                dim=2)  # [B, n_mas, n_nodes]
            ma_emb_out = self.ma_encoding_block(ma_emb, ope_ma_veh_emb, proc_on_off_time)
            veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, offload_trans_time)
            ope_emb_out = self.ope_encoding_block(ope_emb, ma_emb, proc_time)
        elif self.encoder_version in [4, 5, 6, 7, 8]:   # veh-ope-ma 구조
            # ope embedding
            ma_veh_emb = torch.cat([ma_emb, veh_emb], dim=1)    # [B, n_mas+n_vehs, D_emb]
            proc_offload = torch.cat([proc_time, offload_trans_time_OV.transpose(1,2)], dim=2)  # [B, n_jobs, n_mas+n_vehs]
            ope_emb_out = self.ope_encoding_block(ope_emb, ma_veh_emb, proc_offload)
            # ma embedding
            ma_ope_emb = torch.cat([ma_emb, ope_emb], dim=1)    # [B, n_mas+n_opes, D_emb]
            onload_proc = torch.cat([onload_trans_time, proc_time.transpose(1,2)], dim=2)  # [B, n_mas, n_mas+n_opes]
            ma_emb_out = self.ma_encoding_block(ma_emb, ma_ope_emb, onload_proc)
            # veh embedding
            veh_emb_out = self.veh_encoding_block(veh_emb, ope_emb, offload_trans_time_OV)
        elif self.encoder_version == 1:
            
            node_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1) # [B, n_nodes, D_emb]
            batch_size, n_nodes, _ = node_emb.size()
            # edge_emb = torch.zeros(size=(batch_size, n_nodes, n_nodes))
            # edge_emb[:, :n_opes, n_opes:n_opes+n_mas] = proc_time
            # edge_emb[:, n_opes:n_opes+n_mas, :n_opes] = proc_time.transpose(1,2)
            # edge_emb[:, n_opes:n_opes+n_mas, n_opes:n_opes+n_mas] = onload_trans_time
            # edge_emb[:, n_opes+n_mas:, :n_opes] = offload_trans_time_OV
            # edge_emb[:, :n_opes, n_opes+n_mas:] = offload_trans_time_OV.transpose(1,2)
            
            node_emb_out = self.node_encoding_block(node_emb, node_emb)
            ope_emb_out = node_emb_out[:, :n_opes, :]
            ma_emb_out = node_emb_out[:, n_opes:n_opes+n_mas, :]
            veh_emb_out = node_emb_out[:, n_opes+n_mas:, :]
        
        if self.encoder_version == 6:
            node_emb = torch.cat([ope_emb_out, ma_emb_out, veh_emb_out], dim=1) # [B, n_nodes, D_emb]
            
            node_emb_out = self.global_encoding_block(node_emb, node_emb)
            ope_emb_out = node_emb_out[:, :n_opes, :]
            ma_emb_out = node_emb_out[:, n_opes:n_opes+n_mas, :]
            veh_emb_out = node_emb_out[:, n_opes+n_mas:, :]
        elif self.encoder_version == 8:
            ma_emb_out = self.global_ma_encoding_block(ma_emb_out, ma_emb_out)
            veh_emb_out = self.global_veh_encoding_block(veh_emb_out, veh_emb_out)
            ope_emb_out = self.global_ope_encoding_block(ope_emb_out, ope_emb_out)
        
        return ope_emb_out, ma_emb_out, veh_emb_out
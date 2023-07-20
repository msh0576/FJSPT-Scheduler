import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical
from copy import deepcopy

from GTrans_models.layers.graph_transformer_edge_layer import GraphTransformerLayer
from GTrans_models.encoding_block import EncodingBlock_Base
from DHJS_models.encoding_block import SelfEncodingBlock

class TFJSP_Encoder_GTrans(nn.Module):
    def __init__(self, 
                encoder_version,
                **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.encoder_version = encoder_version
        embedding_dim = model_params['embedding_dim']
        
        
        if encoder_version in [1, 2]:
            self.layers = nn.ModuleList([EncoderLayer_Base(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        else:
            raise Exception('encoder version error!')
        
        if encoder_version in [2]:
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
        
        # === local encoding ===
        for layer in self.layers:
            ope_emb, ma_emb, veh_emb, proc_time, onload_trans_time, offload_trans_time_OV = layer(
                ope_emb, ma_emb, veh_emb, 
                proc_time, offload_trans_time, onload_trans_time, 
                offload_trans_time_OV
            )
        
        if self.encoder_version == 2:
            node_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1) # [B, n_nodes, D_emb]
            
            batch_size, n_nodes, _ = node_emb.size()
            global_edge = torch.zeros(size=(batch_size, n_nodes, n_nodes))
            global_edge[:, :n_opes, n_opes:n_opes+n_mas] = proc_time
            global_edge[:, n_opes:n_opes+n_mas, :n_opes] = proc_time.transpose(1,2)
            global_edge[:, n_opes:n_opes+n_mas, n_opes:n_opes+n_mas] = onload_trans_time
            global_edge[:, n_opes+n_mas:, :n_opes] = offload_trans_time_OV
            global_edge[:, :n_opes, n_opes+n_mas:] = offload_trans_time_OV.transpose(1,2)
            
            
            
            node_emb_out, _ = self.global_encoder(node_emb, node_emb, global_edge)
            
            ope_emb = node_emb_out[:, :n_opes, :]
            ma_emb = node_emb_out[:, n_opes:n_opes+n_mas, :]
            veh_emb = node_emb_out[:, n_opes+n_mas:, :]

        return ope_emb, ma_emb, veh_emb, proc_time, onload_trans_time, offload_trans_time_OV


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
        if self.encoder_version == 2:
            self.ma_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
            self.ope_encoding_block = EncodingBlock_Base(**model_params)
        elif self.encoder_version == 1:
            self.node_encoding_block = SelfEncodingBlock(**model_params)
    
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
        
        if self.encoder_version == 2:
            # ope embedding
            ma_veh_emb = torch.cat([ma_emb, veh_emb], dim=1)    # [B, n_mas+n_vehs, D_emb]
            proc_offload = torch.cat([proc_time, offload_trans_time_OV.transpose(1,2)], dim=2)  # [B, n_jobs, n_mas+n_vehs]
            ope_emb_out, proc_offload_out = self.ope_encoding_block(ope_emb, ma_veh_emb, proc_offload)
            proc_time_out1 = proc_offload_out[:, :, :n_mas] # [B, n_jobs, n_mas]
            offload_trans_time_OV_out1 = proc_offload_out[:, :, n_mas:]  # [B, n_jobs, n_vehs]
            # ma embedding
            ma_ope_emb = torch.cat([ma_emb, ope_emb], dim=1)    # [B, n_mas+n_jobs, D_emb]
            onload_proc = torch.cat([onload_trans_time, proc_time.transpose(1,2)], dim=2)  # [B, n_mas, n_mas+n_jobs]
            ma_emb_out, onload_proc_out = self.ma_encoding_block(ma_emb, ma_ope_emb, onload_proc)
            onload_trans_time_out = onload_proc_out[:, :, :n_mas]
            proc_time_out2 = onload_proc_out[:, :, n_mas:]
            # veh embedding
            veh_emb_out, offload_trans_time_OV_out2 = self.veh_encoding_block(veh_emb, ope_emb, offload_trans_time_OV)
            proc_time_out = proc_time_out1 + proc_time_out2.transpose(1,2) # [B, n_jobs, n_mas]
            offload_trans_time_OV_out = offload_trans_time_OV_out1.transpose(1,2) + offload_trans_time_OV_out2  # [B, n_vehs, n_jobs]
        elif self.encoder_version == 1:
            proc_time_out = proc_time
            onload_trans_time_out = onload_trans_time
            offload_trans_time_OV_out = offload_trans_time_OV
            
            node_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1) # [B, n_nodes, D_emb]
            node_emb_out = self.node_encoding_block(node_emb, node_emb)
            ope_emb_out = node_emb_out[:, :n_opes, :]
            ma_emb_out = node_emb_out[:, n_opes:n_opes+n_mas, :]
            veh_emb_out = node_emb_out[:, n_opes+n_mas:, :]
            
        return ope_emb_out, ma_emb_out, veh_emb_out, proc_time_out, onload_trans_time_out, offload_trans_time_OV_out
        
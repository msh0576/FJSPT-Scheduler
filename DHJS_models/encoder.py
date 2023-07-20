from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical
from copy import deepcopy

from DHJS_models.encoding_block import EncodingBlock_Base, EncodingBlock_Job, EncodingBlock_Traj,\
    EncodingBlock_JobAdj, CDN
from DHJS_models.encoder_cdn import EncoderLayer_CDN
from DHJS_models.encoder_veh import EncoderLayer_AugVeh
from DHJS_models.encoder_intent import EncoderLayer_intent


class TFJSP_Encoder_DHJS(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.encoder_version = encoder_version
        if encoder_version in [1, 2, 3, 4, 5]:
            self.layers = nn.ModuleList([EncoderLayer_Base(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        elif encoder_version in [6]:
            self.layers = nn.ModuleList([EncoderLayer_CDN(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        elif encoder_version in [7, 8, 12]:
            self.layers = nn.ModuleList([EncoderLayer_AugVeh(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        elif encoder_version in [9, 10, 11]:
            self.layers = nn.ModuleList([EncoderLayer_intent(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        else:
            raise Exception('encoder version error!')

    def init(self):
        '''
        encoder version 3
        '''
        for layer in self.layers:
            layer.init()
           

    def forward(self, job_emb, ma_emb, veh_emb, proc_time_mat, 
            offload_trans_time, trans_time_mat, 
            oper_adj_batch=None, batch_core_adj_list=None, MVpair_trans_time=None,
            onload_trans_time=None, mask_dyn_ope_ma_adj=None,
            mask_ma=None,
        ):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        for layer in self.layers:
            job_emb, ma_emb, veh_emb = layer(
                job_emb, ma_emb, veh_emb, proc_time_mat, offload_trans_time, 
                trans_time_mat, oper_adj_batch, batch_core_adj_list, MVpair_trans_time,
                onload_trans_time, mask_dyn_ope_ma_adj, mask_ma
            )
        return job_emb, ma_emb, veh_emb

class EncoderLayer_Base(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        
        self.ma_encoding_block = EncodingBlock_Base(**model_params)
        self.veh_encoding_block = EncodingBlock_Base(**model_params)
        if encoder_version in [1, 2, 3, 4, 5]:
            self.job_encoding_block = EncodingBlock_Job(**model_params)
            
            # if encoder_version == 3:
            #     self.encoding_block_traj = EncodingBlock_Traj(bias=True, **model_params)
            #     self.time_length = model_params["time_length"]
            
            if encoder_version == 5:
                self.diffusion_block = CDN(
                    model_params['embedding_dim'], model_params['hidden_dim'], 
                    model_params['embedding_dim'], diffusion_num=2, bias=True, rnn_type='GRU'
                )
        # elif encoder_version == 4:
        #     self.job_encoding_block = EncodingBlock_JobAdj(**model_params)
        else:
            raise Exception('EncoderLayer_JobEnc error!')

    # def init(self):
    #     embedding_dim = self.model_params['embedding_dim']
    #     batch_size = self.model_params['batch_size']
    #     num_opes = self.model_params['num_opes']
    #     num_mas = self.model_params['num_mas']
    #     num_vehs = self.model_params['num_vehs']
                
    #     self.embed_feat_ope_list = torch.zeros(size=(self.time_length, batch_size, num_opes, embedding_dim))
    #     self.embed_feat_ma_list = torch.zeros(size=(self.time_length, batch_size, num_mas, embedding_dim))
    #     self.embed_feat_veh_list = torch.zeros(size=(self.time_length, batch_size, num_vehs, embedding_dim))
    #     self.norm_proc_trans_time_list = torch.zeros(size=(self.time_length, batch_size, num_opes, num_mas))
    #     self.norm_offload_trans_time_list = torch.zeros(size=(self.time_length, batch_size, num_opes, num_vehs))
    #     self.norm_trans_time_list = torch.zeros(size=(self.time_length, batch_size, num_mas, num_mas))

    def forward(self, 
            ope_emb, ma_emb, veh_emb, 
            proc_time, offload_trans_time, trans_time, 
            oper_adj_batch=None, batch_core_adj_list=None, 
            MVpair_trans_time=None, onload_trans_time=None,
            mask_dyn_ope_ma_adj=None, mask_ma=None
        ):
        '''
        :params ope_emb: [B, n_opes, E]
        :params ma_emb: [B, n_mas, E]
        :params veh_emb: [B, n_vehs, E]
        :params proc_time: [B, n_opes, n_mas]
        :params offload_trans_time: [B, n_opes, n_vehs]
        :params trans_time: [B, n_mas, n_mas]
        :params oper_adj_batch: [B, n_opes, n_opes]
        :params batch_core_adj_list: [B, max_kcore, n_nodes, n_nodes]
        :params MVpair_trans_time: [B, n_vehs, n_mas]
        :params onload_trans_time [B, n_opes, n_mas]
        :params mask_dyn_ope_ma_adj [B, n_opes, n_mas]
        :param mask_ma [B, n_mas]
        
        Output:
            ope_emb_out: [B, n_opes, E]
            ma_emb_out: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
        '''
        num_opes = ope_emb.size(1)
        num_mas = ma_emb.size(1)
        num_vehs = veh_emb.size(1)
        
        # if self.encoder_version == 3:
        #     # make embedding trajectory 
        #     # ope_emb_: [T, B, n_opes, D_emb]
        #     tmp_ope_emb, tmp_ma_emb, tmp_veh_emb, \
        #         tmp_proc_time_mat, tmp_empty_trans_time_mat, tmp_trans_time_mat = \
        #         self._embed_list(ope_emb, ma_emb, veh_emb, \
        #         proc_time_mat, empty_trans_time_mat, trans_time_mat)
        #     # RNN
        #     # ope_emb_: [B, n_opes, D_emb]
        #     ope_emb_, ma_emb_, veh_emb_, proc_time_mat_, empty_trans_time_mat_ = \
        #         self.encoding_block_traj(tmp_ope_emb, tmp_ma_emb, tmp_veh_emb, tmp_proc_time_mat, tmp_empty_trans_time_mat)
        
            
        ope_emb_out = self.job_encoding_block(ope_emb, ma_emb, veh_emb, proc_time, offload_trans_time, oper_adj_batch)   # [B, n_jobs, E]
        ma_emb_out = self.ma_encoding_block(ma_emb, ope_emb, proc_time.transpose(1, 2))   # [B, n_mas, E]
        veh_emb_out = self.veh_encoding_block(veh_emb, ope_emb, offload_trans_time.transpose(1, 2)) # [B, n_vehs, E]
        
        if self.encoder_version == 5:
            node_emb = torch.cat([ope_emb_out, ma_emb_out, veh_emb_out], dim=1).float() # [B, n_nodes, D_emb]
            
            node_emb_out = self.diffusion_block(node_emb, batch_core_adj_list)
            ope_emb_out = node_emb_out[:, :num_opes, :]
            ma_emb_out = node_emb_out[:, num_opes:num_opes+num_mas, :]
            veh_emb_out = node_emb_out[:, num_opes+num_mas:num_opes+num_mas+num_vehs, :]
        
        # print(f'ope_emb_out:{ope_emb_out}')
        return ope_emb_out, ma_emb_out, veh_emb_out

    def _embed_list(self,
        embed_feat_ope, embed_feat_ma, embed_feat_veh,
        norm_proc_trans_time, norm_offload_trans_time, norm_trans_time
    ):
        '''
        Time shift, and insert current embed_feat
        '''
        
        self.embed_feat_ope_list = self.embed_feat_ope_list.roll(1,0)
        self.embed_feat_ope_list[-1] = embed_feat_ope
        self.embed_feat_ma_list = self.embed_feat_ma_list.roll(1,0)
        self.embed_feat_ma_list[-1] = embed_feat_ma
        self.embed_feat_veh_list = self.embed_feat_veh_list.roll(1,0)
        self.embed_feat_veh_list[-1] = embed_feat_veh
        self.norm_proc_trans_time_list = self.norm_proc_trans_time_list.roll(1,0)
        self.norm_proc_trans_time_list[-1] = norm_proc_trans_time
        self.norm_offload_trans_time_list = self.norm_offload_trans_time_list.roll(1,0)
        self.norm_offload_trans_time_list[-1] = norm_offload_trans_time
        self.norm_trans_time_list = self.norm_trans_time_list.roll(1,0)
        self.norm_trans_time_list[-1] = norm_trans_time
        
        
        
        # self.embed_feat_ope_list.append(embed_feat_ope)
        # self.embed_feat_ma_list.append(embed_feat_ma)
        # self.embed_feat_veh_list.append(embed_feat_veh)
        # self.norm_proc_trans_time_list.append(norm_proc_trans_time)
        # self.norm_offload_trans_time_list.append(norm_offload_trans_time)
        # self.norm_trans_time_list.append(norm_trans_time)
        
        # if len(self.embed_feat_ope_list) > self.time_length:
        #     self.embed_feat_ope_list = self.embed_feat_ope_list[-self.time_length:]
        #     self.embed_feat_ma_list = self.embed_feat_ma_list[-self.time_length:]
        #     self.embed_feat_veh = self.embed_feat_veh[-self.time_length:]
        #     self.norm_proc_trans_time_list = self.norm_proc_trans_time_list[-self.time_length:]
        #     self.norm_offload_trans_time_list = self.norm_offload_trans_time_list[-self.time_length:]
        #     self.norm_trans_time_list = self.norm_trans_time_list[-self.time_length:]
        # else:
        #     resd_len = self.time_length - len(self.embed_feat_ope_list)
        #     self.embed_feat_ope_list = [self.embed_feat_ope_list[0]] * resd_len + self.embed_feat_ope_list
        #     self.embed_feat_ma_list = [self.embed_feat_ma_list[0]] * resd_len + self.embed_feat_ma_list
        #     self.embed_feat_veh = [self.embed_feat_veh[0]] * resd_len + self.embed_feat_veh
        #     self.norm_proc_trans_time_list = [self.norm_proc_trans_time_list[0]] * resd_len + self.norm_proc_trans_time_list
        #     self.norm_offload_trans_time_list = [self.norm_offload_trans_time_list[0]] * resd_len + self.norm_offload_trans_time_list
        #     self.norm_trans_time_list = [self.norm_trans_time_list[0]] * resd_len + self.norm_trans_time_list
            
        # return torch.stack(self.embed_feat_ope_list), torch.stack(self.embed_feat_ma_list), \
        #     torch.stack(self.embed_feat_veh), torch.stack(self.norm_proc_trans_time_list), \
        #         torch.stack(self.norm_offload_trans_time_list), torch.stack(self.norm_trans_time_list)
        return self.embed_feat_ope_list[:], self.embed_feat_ma_list[:], self.embed_feat_veh_list[:],\
            self.norm_proc_trans_time_list[:], self.norm_offload_trans_time_list[:],\
                self.norm_trans_time_list[:]
                
class EncoderLayer_allNoes(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        
        if encoder_version == 6:
            # 하나의 encoding block에서 모든 노드 처리
            # 생각보다 잘됨...
            self.encoding_block = EncodingBlock_Base(**model_params)
        else:
            raise Exception('encoder version error!')
        
    
    def forward(self, ope_emb, ma_emb, veh_emb, proc_time_mat, empty_trans_time_mat, trans_time_mat, 
                oper_adj_batch=None, batch_core_adj_list=None):
        '''
        Input:
            ope_emb: [B, n_opes, E]
            ma_emb: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
            proc_time_mat: [B, n_opes, n_mas]
            empty_trans_time_mat: [B, n_opes, n_vehs]
            trans_time_mat: [B, n_mas, n_mas]
            oper_adj_batch: [B, n_opes, n_opes]
            batch_core_adj_list: [B, max_kcore, n_nodes, n_nodes]
        Output:
            ope_emb_out: [B, n_opes, E]
            ma_emb_out: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
        '''
        batch_size, num_opes, D_emb = ope_emb.size()
        num_mas = ma_emb.size(1)
        num_vehs = veh_emb.size(1)
        
        nodes_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1)    # [B, n_nodes, D_emb]
        ope_adj = torch.cat([oper_adj_batch, proc_time_mat, empty_trans_time_mat], dim=-1)    # [B, n_opes, n_opes+n_mas+n_vehs]
        proc_time_mat_trans = proc_time_mat.transpose(1, 2)    # [B, n_mas, n_opes]
        zero_ma_adj = torch.zeros(size=(batch_size, num_mas, num_mas+num_vehs))
        ma_adj = torch.cat([proc_time_mat_trans, zero_ma_adj], dim=-1) # [B, n_mas, n_opes+n_mas+n_vehs]
        
        empty_trans_time_mat_trans = empty_trans_time_mat.transpose(1, 2)   # [B, n_vehs, n_opes]
        zero_veh_adj = torch.zeros(size=(batch_size, num_vehs, num_mas+num_vehs))
        veh_adj = torch.cat([empty_trans_time_mat_trans, zero_veh_adj], dim=-1)  # [B, n_vehs, n_opes+n_mas+n_vehs]
        
        nodes_adj = torch.cat([ope_adj, ma_adj, veh_adj], dim=1)    # [B, n_opes+n_mas+n_vehs, n_opes+n_mas+n_vehs]
        
        
        nodes_emb_out = self.encoding_block(nodes_emb, nodes_emb, nodes_adj) # [B, n_nodes, E]
        
        ope_emb_out = nodes_emb_out[:, :num_opes, :]
        ma_emb_out = nodes_emb_out[:, num_opes:num_opes+num_mas, :]
        veh_emb_out = nodes_emb_out[:, num_opes+num_mas:num_opes+num_mas+num_vehs, :]
        return ope_emb_out, ma_emb_out, veh_emb_out
        
        
        
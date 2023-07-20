from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from env.common_func import reshape_by_heads

from DHJS_models.encoding_block import EncodingBlock_Base, EncodingBlock_Job, SelfEncodingBlock
from matnet_models.FFSPModel_SUB import AddAndInstanceNormalization, FeedForward, MixedScore_MultiHeadAttention



class EncoderLayer_AugVeh(nn.Module):
    '''
    encoder version 7, 8, 12
    '''
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        self.job_centric = model_params['job_centric']
        
        if encoder_version == 7:
            self.ma_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
            # self.veh_encoding_block = EncodingBlock_Base(**model_params)
            self.job_encoding_block = EncodingBlock_Job(**model_params)
        elif encoder_version == 8:
            self.ma_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
            self.job_encoding_block = EncodingBlock_Base(**model_params)
        elif encoder_version == 12:
            self.ma_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
            self.job_encoding_block = EncodingBlock_Base(**model_params)
            self.total_encoding_block = SelfEncodingBlock(**model_params)
        else:
            raise Exception('encoder_version error!')
    
    def forward(self, 
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
        
        if self.encoder_version == 7:
            proc_trans_time = torch.cat([proc_time.transpose(1, 2), trans_time], dim=-1)    # [B, n_mas, n_opes+n_mas]
            ope_ma_emb = torch.cat([ope_emb, ma_emb], dim=1)    # [B, n_opes+n_mas, D_emb]
            ma_emb_out = self.ma_encoding_block(ma_emb, ope_ma_emb, proc_trans_time)   # [B, n_mas, E]
            veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, MVpair_trans)   # [B, n_vehs, E]
            ope_emb_out = self.job_encoding_block(ope_emb, ma_emb, veh_emb, proc_time, offload_trans_time)   # [B, n_jobs, E]
        # 남은 정보를 embedding 할 순 없나?
        elif self.encoder_version == 8:
            # [B, n_mas, n_opes+n_mas+n_vehs]
            proc_OnOffTrans_time = torch.cat([proc_time.transpose(1, 2), trans_time, MVpair_trans.transpose(1, 2)], dim=-1)   
            ope_ma_veh_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1)    # [B, n_opes+n_mas_n_vehs, D_emb]
            ma_emb_out = self.ma_encoding_block(ma_emb, ope_ma_veh_emb, proc_OnOffTrans_time)   # [B, n_mas, E]
            veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, MVpair_trans)   # [B, n_vehs, E]
            ope_emb_out = self.job_encoding_block(ope_emb, ma_emb, proc_time)   # [B, n_jobs, E]
        elif self.encoder_version == 12:
            # [B, n_mas, n_opes+n_mas+n_vehs]
            proc_OnOffTrans_time = torch.cat([proc_time.transpose(1, 2), trans_time, MVpair_trans.transpose(1, 2)], dim=-1)   
            ope_ma_veh_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1)    # [B, n_opes+n_mas_n_vehs, D_emb]
            tmp_ma_emb_out = self.ma_encoding_block(ma_emb, ope_ma_veh_emb, proc_OnOffTrans_time)   # [B, n_mas, E]
            tmp_veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, MVpair_trans)   # [B, n_vehs, E]
            tmp_ope_emb_out = self.job_encoding_block(ope_emb, ma_emb, proc_time)   # [B, n_jobs, E]
            
            total_emb = torch.cat([tmp_ope_emb_out, tmp_ma_emb_out, tmp_veh_emb_out], dim=1)    # [B, n_node, D_emb]
            total_emb_out = self.total_encoding_block(total_emb, total_emb) # [B, n_node, D_emb]
            
            ope_emb_out = total_emb_out[:, :num_opes, :]
            ma_emb_out = total_emb_out[:, num_opes:num_opes+num_mas, :]
            veh_emb_out = total_emb_out[:, num_opes+num_mas:, :]
        
        return ope_emb_out, ma_emb_out, veh_emb_out
        
        
        
class EncodingBlock_veh(EncodingBlock_Base):
    def __init__(self, **model_params):
        super().__init__(**model_params)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        self.Wq_ma = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_ma = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_ma = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA_ma = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine_ma = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, veh_emb, ma_emb, MVpair_trans_time):
        '''
        :param veh_emb [B, n_vehs, E]:
        :param ma_emb [B, n_mas, E]:
        :param MVpair_trans_time [B, n_vehs, n_mas]:
        :param trans_time: [B, n_mas, n_mas]
        '''
        head_num = self.model_params['head_num']

        # === vehicle-machine embedding ===
        q = reshape_by_heads(self.Wq(veh_emb), head_num=head_num)
        k = reshape_by_heads(self.Wk(ma_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(ma_emb), head_num=head_num)
        
        out_concat = self.mixed_score_MHA(q, k, v, MVpair_trans_time)
        # shape: (batch, row_cnt, head_num*qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)
        
        
        out1 = self.add_n_normalization_1(veh_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        
        return out3

class EncodingBlock_Job(EncodingBlock_Base):
    '''
    encoder version = 1, 2
    '''
    def __init__(self, **model_params):
        super().__init__(**model_params)

    def forward(self, ope_emb, ma_emb, veh_emb, proc_time_mat, trans_time_mat, ope_adj_mat=None):
        '''
        Input:
            ope_emb: [B, n_opes, E]
            ma_emb: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
            proc_time_mat: [B, n_opes, n_mas]
            trans_time_mat: [B, n_opes, n_vehs]
        '''
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']
        
        ma_veh_emb = torch.cat([ma_emb, veh_emb], dim=1)    # [B, n_mas + n_vehs, E]
        proc_trans_time_mat = torch.cat([proc_time_mat, trans_time_mat], dim=-1)    # [B, n_opes, n_mas + n_vehs]

        q = reshape_by_heads(self.Wq(ope_emb), head_num=head_num)   # [B, H, n_opes, qkv_dim]

        k = reshape_by_heads(self.Wk(ma_veh_emb), head_num=head_num) # [B, H, n_mas, qkv_dim]
        v = reshape_by_heads(self.Wv(ma_veh_emb), head_num=head_num)

        out_concat = self.mixed_score_MHA(q, k, v, proc_trans_time_mat)  # [B, n_opes, H*qkv_dim]
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(ope_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
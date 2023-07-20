
from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from env.common_func import reshape_by_heads

from DHJS_models.encoding_block import EncodingBlock_Base, EncodingBlock_Job
from matnet_models.FFSPModel_SUB import AddAndInstanceNormalization, FeedForward, MixedScore_MultiHeadAttention


class EncoderLayer_intent(nn.Module):
    '''
    encoder version 9, 10, 11
    '''
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.transtime_btw_ma_max = model_params["transtime_btw_ma_max"]
        
        if encoder_version == 9:
            self.ope_encoding_block = EncodingBlock_Base(**model_params)
            self.ma_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
        elif encoder_version == 10:
            self.ope_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
            self.ma1_encoding_block = EncodingBlock_Base(**model_params)
            self.ma2_encoding_block = EncodingBlock_Base(**model_params)
            self.ma_proj = nn.Linear(2*self.embedding_dim, self.embedding_dim)
        elif encoder_version == 11:
            self.ope_encoding_block = EncodingBlock_Base(**model_params)
            self.veh_encoding_block = EncodingBlock_Base(**model_params)
            self.ma1_encoding_block = EncodingBlock_Base(**model_params)
            self.ma2_encoding_block = EncodingBlock_Base(**model_params)
            self.ma_proj = nn.Linear(2*self.embedding_dim, self.embedding_dim)
            
            self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim, num_layers=1, bias=True, batch_first=True)
            self.norm = nn.LayerNorm(self.embedding_dim)
        else:
            raise Exception('encoder_version error!')
        self.num_intent = 4
        
    def init(self):
        pass        

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
        :param proc_time: [B, n_opes, n_mas]
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
        batch_size, num_opes, _ = ope_emb.size()
        num_mas = ma_emb.size(1)
        num_vehs = veh_emb.size(1)
        num_nodes = num_opes + num_mas + num_vehs
        
        
        # === operation intent: 가장 빠른 processing machine만 embedding ===
        #: 남은 operations 중에서 가장 processing time 이 짧은 머신 하나만 선택
        #: 특정 ope는 연결된 ma 가 없을수도 있음
        A_proc_time = torch.zeros_like(proc_time)
        A_proc_time[mask_dyn_ope_ma_adj] = proc_time[mask_dyn_ope_ma_adj]
        A_proc_time = self.min_ProcTime_elements(A_proc_time)   # [B, n_opes, n_mas]
        A_proc_time_norm = torch.where(A_proc_time>0, 1., 0.)
        
        # === machine intent1: 최대한 많은 ma이 동시에 동작하도록 ===
        #: 이걸 표현하기 쉽지 않음...
        #: 사용중인 머신들간 adj_mat 생성
        tmp_row_mask = mask_ma[:,None, :].expand(-1, mask_ma.size(1), -1)
        tmp_col_mask = mask_ma[:, :, None].expand(-1, -1, mask_ma.size(1))
        A_nonprocing_ma_mask = (tmp_row_mask == True) & (tmp_col_mask == True)  # [B, n_mas, n_mas]
        A_procing_ma = (~A_nonprocing_ma_mask).float()
        
        # === vehicle intent: 가장 가까운 머신 ===
        A_offloadTrans_time_norm = self.inverse_normalize_matrix(MVpair_trans)  # [B, n_vehs, n_mas]
        # === machine intent: 머신 사이의 이동거리 among eligible mas ===
        trans_time[~A_nonprocing_ma_mask] = self.transtime_btw_ma_max+1
        A_onloadTrans_time_norm = self.inverse_normalize_matrix(trans_time)  # [B, n_mas, n_mas]
        # === incorporate A ===
        total_A = torch.zeros(size=(batch_size, num_nodes, num_nodes))
        
        total_A[:, :num_opes, num_opes:num_opes+num_mas] = A_proc_time_norm
        total_A[:, num_opes:num_opes+num_mas, num_opes:num_opes+num_mas] = A_procing_ma
        total_A[:, num_opes+num_mas:, num_opes:num_opes+num_mas] = A_offloadTrans_time_norm
        total_A[:, num_opes:num_opes+num_mas, num_opes:num_opes+num_mas] += A_onloadTrans_time_norm
        
        # === encoding ===
        if self.encoder_version == 9:
            ope_emb_out = self.ope_encoding_block(ope_emb, ma_emb, total_A[:, :num_opes, num_opes:num_opes+num_mas])
            ma_emb_out = self.ma_encoding_block(ma_emb, ma_emb, total_A[:, num_opes:num_opes+num_mas, num_opes:num_opes+num_mas])
            veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, total_A[:, num_opes+num_mas:, num_opes:num_opes+num_mas])
        elif self.encoder_version == 10:
            ope_emb_out = self.ope_encoding_block(ope_emb, ma_emb, A_proc_time_norm)
            veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, A_offloadTrans_time_norm)
            ma_emb_out1 = self.ma1_encoding_block(ma_emb, ma_emb, A_procing_ma)
            ma_emb_out2 = self.ma2_encoding_block(ma_emb, ma_emb, A_onloadTrans_time_norm)
            ma_emb_out = self.ma_proj(torch.cat([ma_emb_out1, ma_emb_out2], dim=-1))
        elif self.encoder_version == 11:
            tmp_ope_emb_out = self.ope_encoding_block(ope_emb, ma_emb, A_proc_time_norm)
            tmp_veh_emb_out = self.veh_encoding_block(veh_emb, ma_emb, A_offloadTrans_time_norm)
            tmp_ma_emb_out1 = self.ma1_encoding_block(ma_emb, ma_emb, A_procing_ma)
            tmp_ma_emb_out2 = self.ma2_encoding_block(ma_emb, ma_emb, A_onloadTrans_time_norm)
                
            ope_emb_padd = self._zero_pad_matrix(tmp_ope_emb_out, num_nodes-num_opes)    # [B, num_nodes, D_emb]
            ma1_emb_padd = self._zero_pad_matrix(tmp_ma_emb_out1, num_nodes-num_mas)
            ma2_emb_padd = self._zero_pad_matrix(tmp_ma_emb_out2, num_nodes-num_mas)
            veh_emb_padd = self._zero_pad_matrix(tmp_veh_emb_out, num_nodes-num_vehs)
            
            emb_stack = torch.stack([ope_emb_padd, ma1_emb_padd, ma2_emb_padd, veh_emb_padd])    # [4, B, num_nodes, D_emb]
            emb_stack_trans = emb_stack.permute(1, 2, 0, 3).reshape(
                batch_size*num_nodes, self.num_intent, self.embedding_dim
            )   # [B*num_nodes, n_intent, D_emb]
            self.rnn.flatten_parameters()
            node_emb_out, _ = self.rnn(emb_stack_trans) # [B*num_nodes, n_intent, D_emb]
            node_emb_out = node_emb_out.sum(dim=1)  # [B*num_nodes, D_output]
            node_emb_out = self.norm(node_emb_out).reshape(batch_size, num_nodes, -1)   # [B, n_nodes, D_emb]
            
            ope_emb_out = node_emb_out[:, :num_opes, :]
            ma_emb_out = node_emb_out[:, num_opes:num_opes+num_mas, :]
            veh_emb_out = node_emb_out[:, num_opes+num_mas:, :]
            
            
            
        return ope_emb_out, ma_emb_out, veh_emb_out
        
    
    def _zero_pad_matrix(self, input_matrix, padding_size):
        B, N, E = input_matrix.size()
        padded_matrix = torch.zeros((B, N + padding_size, E), dtype=input_matrix.dtype)
        padded_matrix[:, :N, :] = input_matrix
        return padded_matrix
        
        
        
    def _get_degree(self, A, num_nodes):
        '''
        :param A [B, row, col]
        
        :return Laplacian_matrix
        '''
        A_scores = F.softmax(A, 1)
        A_in_shape = A_scores.tocoo().shape
        # print(f'A_scores:{A_scores, A_scores.shape}')
        A_indices = np.mat([list(range(num_nodes)), list(range(num_nodes))]).transpose()
        D_indices = np.mat([list(range(num_nodes)), list(range(num_nodes))]).transpose()
        num_intent = A.size(1)
        for k in range(num_intent):
            A_k_scores = A_scores[k]
            A_k_tensor = torch.sparse_coo_tensor(A_indices, A_k_scores, [num_nodes, num_nodes])
            # D_k_col
        
        
    def min_ProcTime_elements(self, tensor):
        '''
        for row dimension, keep only minimum value and others change to 0
        :param tensor: [B, col, row]
        '''
        non_zeros = tensor.clone()
        non_zeros[non_zeros == 0] = float('inf')
        min_values, _ = torch.min(non_zeros, dim=2, keepdim=True)
        min_values_mask = (tensor == min_values) & (tensor != 0)
        output = tensor * min_values_mask.to(torch.float32)
        return output
    
    def normalize_matrix(self, input_matrix):
        '''
        normalize 2D matrix [col, row]
        :param input_matrix [B, col, row]
        '''
        min_values, _ = torch.min(input_matrix.view(input_matrix.size(0), -1), dim=1, keepdim=True)
        max_values, _ = torch.max(input_matrix.view(input_matrix.size(0), -1), dim=1, keepdim=True)
        
        min_values = min_values.view(input_matrix.size(0), 1, 1)
        max_values = max_values.view(input_matrix.size(0), 1, 1)

        output_matrix = (input_matrix - min_values) / (max_values - min_values)
        return output_matrix

    def inverse_normalize_matrix(self, input_matrix):
        norm_matrix = self.normalize_matrix(input_matrix)
        return 1 - norm_matrix
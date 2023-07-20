
import torch
import torch.nn as nn

from env.common_func import reshape_by_heads

from matnet_models.FFSPModel_SUB import AddAndInstanceNormalization, \
    FeedForward, MixedScore_MultiHeadAttention, MultiHeadAttention

class EncodingBlock_Base(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)

class SelfEncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.MHA = MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.MHA(q, k, v)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
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

class EncodingBlock_JobAdj(EncodingBlock_Base):
    '''
    encoder version = 4
    '''
    def __init__(self, **model_params):
        super().__init__(**model_params)

    def forward(self, ope_emb, ma_emb, veh_emb, proc_time_mat, trans_time_mat, ope_adj_mat):
        '''
        Input:
            ope_emb: [B, n_jobs, E]
            ma_emb: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
            proc_time_mat: [B, n_jobs, n_mas]
            trans_time_mat: [B, n_jobs, n_vehs]
            ope_adj_mat: [B, n_opes, n_opes]
        '''
        head_num = self.model_params['head_num']
        ope_ma_veh_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1)    # [B, n_opes + n_mas + n_vehs, E]
        opeAdj_proc_trans_time_mat = torch.cat([ope_adj_mat, proc_time_mat, trans_time_mat], dim=-1)    # [B, n_opes, n_opes + n_mas + n_vehs]

        q = reshape_by_heads(self.Wq(ope_emb), head_num=head_num)   # [B, H, n_opes, qkv_dim]

        k = reshape_by_heads(self.Wk(ope_ma_veh_emb), head_num=head_num) # [B, H, n_mas, qkv_dim]
        v = reshape_by_heads(self.Wv(ope_ma_veh_emb), head_num=head_num)

        out_concat = self.mixed_score_MHA(q, k, v, opeAdj_proc_trans_time_mat)  # [B, n_opes, H*qkv_dim]
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(ope_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
    

class EncodingBlock_Traj(nn.Module):
    '''
    '''
    def __init__(self, bias=True, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.rnn_ope = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, bias=bias, batch_first=True)
        self.rnn_ma = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, bias=bias, batch_first=True)
        self.rnn_veh = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, bias=bias, batch_first=True)
        self.rnn_proc_time = nn.GRU(input_size=1, hidden_size=1, num_layers=1, bias=bias, batch_first=True)
        self.rnn_trans_time = nn.GRU(input_size=1, hidden_size=1, num_layers=1, bias=bias, batch_first=True)
    
    
    def forward(self, ope_emb, ma_emb, veh_emb, proc_time_mat, trans_time_mat):
        '''
        Input:
            ope_emb: [T, B, n_jobs, E]
            ma_emb: [T, B, n_mas, E]
            veh_emb: [T, B, n_vehs, E]
            proc_time_mat: [T, B, n_jobs, n_mas]
            trans_time_mat: [T, B, n_jobs, n_vehs]
        Output:
            output_ope: [B, n_opes, D_emb]
            output_ma: [B, n_mas, D_emb]
            output_veh: [B, n_vehs, D_emb]
            output_proc_time: [B, n_opes, n_mas]
            output_trans_time: [B, n_opes, n_vehs]
        '''
        time_len, batch_size, num_opes, D_emb = ope_emb.size()
        _, _, num_mas, _ = ma_emb.size()
        _, _, num_vehs, _ = veh_emb.size()
        
        ope_emb = ope_emb.permute(1, 2, 0, 3).reshape(batch_size*num_opes, time_len, D_emb)
        ma_emb = ma_emb.permute(1, 2, 0, 3).reshape(batch_size*num_mas, time_len, D_emb)
        veh_emb = veh_emb.permute(1, 2, 0, 3).reshape(batch_size*num_vehs, time_len, D_emb)
        
        proc_time_mat = proc_time_mat.unsqueeze(-1).permute(1, 2, 3, 0, 4).reshape(batch_size*num_opes*num_mas, time_len, 1)
        trans_time_mat = trans_time_mat.unsqueeze(-1).permute(1, 2, 3, 0, 4).reshape(batch_size*num_opes*num_vehs, time_len, 1)
        
        self.rnn_ope.flatten_parameters()
        self.rnn_ma.flatten_parameters()
        self.rnn_veh.flatten_parameters()
        self.rnn_proc_time.flatten_parameters()
        self.rnn_trans_time.flatten_parameters()
        output_ope, _ = self.rnn_ope(ope_emb)
        output_ma, _ = self.rnn_ma(ma_emb)
        output_veh, _ = self.rnn_veh(veh_emb)
        output_proc_time, _ = self.rnn_proc_time(proc_time_mat)
        output_trans_time, _ = self.rnn_trans_time(trans_time_mat)
        
        output_ope = output_ope.reshape(batch_size, num_opes, time_len, D_emb)[:, :, 0, :]
        output_ma = output_ma.reshape(batch_size, num_mas, time_len, D_emb)[:, :, 0, :]
        output_veh = output_veh.reshape(batch_size, num_vehs, time_len, D_emb)[:, :, 0, :]
        output_proc_time = output_proc_time.reshape(batch_size, num_opes, num_mas, time_len, 1)[:, :, :, 0, :].squeeze(-1)
        output_trans_time = output_trans_time.reshape(batch_size, num_opes, num_vehs, time_len, 1)[:, :, :, 0, :].squeeze(-1)
        
        
        return output_ope, output_ma, output_veh, output_proc_time, output_trans_time


from DHJS_models.layers import CoreDiffusion, CoreDiffusionBatch
# K-core diffusion netowrk
class CDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, diffusion_num, bias=True, rnn_type='GRU'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type

        if diffusion_num == 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusionBatch(input_dim, output_dim, bias=bias, rnn_type=rnn_type))
        elif diffusion_num > 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusionBatch(input_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            for i in range(diffusion_num - 2):
                self.diffusion_list.append(CoreDiffusionBatch(hidden_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            self.diffusion_list.append(CoreDiffusionBatch(hidden_dim, output_dim, bias=bias, rnn_type=rnn_type))
        else:
            raise ValueError("number of layers should be positive!")
    
    def forward(self, x, adj_list):
        '''
        Input:
            x: [B, n_nodes, embed_dim]
            adj_list: [B, max_core, n_nodes, n_nodes]
        '''
        for i in range(self.diffusion_num):
            x = self.diffusion_list[i](x, adj_list)
        return x
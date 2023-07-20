
import torch
import torch.nn as nn

from env.common_func import reshape_by_heads

from matnet_models.FFSPModel_SUB import AddAndInstanceNormalization, AddAndInstanceNormalization_Edge, \
    FeedForward, FeedForward_Edge, MixedScore_MultiHeadAttention, MultiHeadAttention, MixedScore_MultiHeadAttention_WithEdge


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
        self.mixed_score_MHA = MixedScore_MultiHeadAttention_WithEdge(**model_params)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)
        
        self.multi_head_combine_edge = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_n_normalization_1_edge = AddAndInstanceNormalization_Edge(**model_params)
        self.feed_forward_edge = FeedForward_Edge(**model_params)
        self.add_n_normalization_2_edge = AddAndInstanceNormalization_Edge(**model_params)

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

        out_concat, out_edge = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim) | (B, H, row_cnt, col_cnt)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        
        multi_head_out_edge = out_edge.sum(dim=1).unsqueeze(-1)  # [B, row_cnt, col_cnt, 1]
        batch_size, row_cnt, col_cnt, _ = multi_head_out_edge.size()
        multi_head_out_edge_resh = multi_head_out_edge.reshape(batch_size, row_cnt*col_cnt, 1)
        cost_mat_resh = cost_mat.unsqueeze(-1).reshape(batch_size, row_cnt*col_cnt, 1)
        out1_edge = self.add_n_normalization_1_edge(cost_mat_resh, multi_head_out_edge_resh)
        out2_edge = self.feed_forward_edge(out1_edge)
        out3_edge = self.add_n_normalization_2_edge(out1_edge, out2_edge)
        out3_edge = out3_edge.reshape(batch_size, row_cnt, col_cnt)
        # print(f'out3_edge:{out3_edge.shape}')
        return out3, out3_edge
        # shape: (batch, row_cnt, embedding)
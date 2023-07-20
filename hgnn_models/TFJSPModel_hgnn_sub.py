import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F

class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        # print(f"W_sizes_ope:{W_sizes_ope} | hidden_size_ope:{hidden_size_ope} | out_size_ope:{out_size_ope}")
        
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        h = (feats[1], feats[0], feats[0], feats[0])
        # print(f"ope_pre_adj_batch:{ope_pre_adj_batch.shape}")
        
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])  # [B, n_opes, n_opes]
        # print(f"self_adj:{self_adj.shape}")
        # Calculate an return operation embedding
        adj = (ope_ma_adj_batch[batch_idxes], ope_pre_adj_batch[batch_idxes],
               ope_sub_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):   
            # print(f"i:{i} -- {self.gnn_layers[i](h[i], adj[i]).shape}")   # [D_b, max(n_ope), 8(D_out)]
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)    # [D_b, max(n_ope), 8*4]
        mu_ij_prime = self.project(MLP_embedding_in)    # [D_b, max(n_ope), 8]
        return mu_ij_prime


class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                1, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, ope_ma_adj_batch, batch_idxes, feat):
        '''
        Input:
            feat: (raw_opes, raw_mas, proc_time)
        '''
        
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            if not hasattr(self, 'fc_src'):
                self.fc_src, self.fc_dst = self.fc, self.fc
            feat_src = self.fc_src(h_src)   # [B, n_opes, out_size_ma] at GATedge(0)
            feat_dst = self.fc_dst(h_dst)   # [B, n_mas, out_size_ma] at GATedge(0)
        else:
            # Deprecated in final experiment
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        
        feat_edge = self.fc_edge(feat[2].unsqueeze(-1)) # [B, n_opes, n_mas, out_size_ma]

        # Calculate attention coefficients
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1) # [B, n_opes, 1]
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) # [B, n_mas, 1]
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)    # [B, n_opes, n_mas, 1]
        el_add_ee = ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(a)
        ekk = self.leaky_relu(er + er)
        # print(f"el:{el.shape} | er:{er.shape} | ee:{ee.shape} | ")

        # Normalize attention coefficients
        mask = torch.cat((ope_ma_adj_batch[batch_idxes].unsqueeze(-1)==1,
                          torch.full(size=(ope_ma_adj_batch[batch_idxes].size(0), 1,
                                           ope_ma_adj_batch[batch_idxes].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)    # [B, n_opes + 1, n_mas, 1]
        e[~mask] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)    # [B, n_opes + 1, n_mas]
        alpha_ijk = alpha[..., :-1, :]  # [B, n_opes, n_mas]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)  # [B, 1, n_mas]

        # Calculate an return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)    # [B, n_opes, n_mas, out_size_ma]
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)
        nu_k_prime = torch.sigmoid(b+c)
        return nu_k_prime

class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats
        # print(f"in_feats:{in_feats} | out_feats:{out_feats} | hidden_dim:{hidden_dim}")

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        # print(f"adj: {adj.shape} | adj.unsqueeze(-1):{adj.unsqueeze(-1).shape} | feat: {feat.shape} | feat.unsqueeze(-3):{feat.unsqueeze(-3).shape}")
        # ex) adj: [B, n_opes, n_opes] | feat: [B, n_opes, out_size_ma]
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)  # [B, n_opes, n_opes, out_size_ma]
        # print(f"a:{a.shape}")
        b = torch.sum(a, dim=-2)    # [B, n_opes, out_size_ma]
        c = self.project(b) # [B, n_opes, out_size_ma]
        # print(f"b:{b.shape} | c:{c.shape}")
        return c

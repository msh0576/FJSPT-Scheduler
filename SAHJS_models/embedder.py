
import torch
import torch.nn as nn
from copy import deepcopy

from env.common_func import generate_trans_mat

class SAHJS_embedder(nn.Module):
    def __init__(
            self,
            embedding_dim_,
            ope_feat_dim,
            ma_feat_dim,
            veh_feat_dim,
            **model_paras,
        ):
        super().__init__()
        self.init_embed_opes = nn.Linear(ope_feat_dim, embedding_dim_)
        self.init_embed_mas = nn.Linear(ma_feat_dim, embedding_dim_)
        self.init_embed_vehs = nn.Linear(veh_feat_dim, embedding_dim_)
        

    def embedding(self, state,  encoder_version=0):
        '''
        Input:
            job_embedding: if True, we embed operation, otherwise job
        
        :return embed_feat_node: [B, n_node, D_emb]
        :return proc_time: [B, n_opes, n_mas]
        :return trans_time: [B, n_mas, n_mas]
        :return offload_trans_time: [B, n_mas, n_vehs]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_finish_batch.size(1)
        num_vehs = state.mask_veh_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        ope_ma_adj_batch = deepcopy(state.ope_ma_adj_batch)
        ope_ma_adj_batch = torch.where(ope_ma_adj_batch == 1, True, False)
        raw_opes = deepcopy(state.feat_opes_batch.transpose(1, 2)[batch_idxes])   # [B, n_opes, ope_feat_dim]
        raw_jobs = raw_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, raw_opes.size(2)))    # [B, n_jobs, ope_feat_dim]
        raw_mas = deepcopy(state.feat_mas_batch.transpose(1, 2)[batch_idxes]) # [B, n_mas, ma_feat_dim]
        raw_vehs = deepcopy(state.feat_vehs_batch.transpose(1, 2)[batch_idxes])   # [B, n_vehs, veh_feat_dim]
        proc_time = deepcopy(state.proc_times_batch[batch_idxes]) # [B, n_opes, n_mas]
        trans_time = deepcopy(state.trans_times_batch[batch_idxes])   # [B, n_mas, n_mas]
        proc_time_on_job = proc_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, proc_time.size(2)))    # [B, n_jobs, n_mas]
        
        # ===
        # [B, n_opes or n_jobs, n_mas, n_vehs] | [B, n_opes, n_vehs] | 
        # [B, n_opes, n_mas] | [B, n_vehs, n_mas]
        _, _, _, offload_trans_time = \
            generate_trans_mat(trans_time, state, job_embedding=False)    
        offload_trans_time = offload_trans_time.transpose(1,2)  # [B, n_mas, n_vehs]
        
        features = self.get_normalized(raw_opes, raw_mas, raw_vehs)
        norm_opes = features[0]
        norm_mas = features[1]
        norm_vehs = features[2]
        # === embeds wrt n_opes, n_mas and n_vehs ===
        embed_feat_ope = self.init_embed_opes(norm_opes)    # [B, n_opes, D_emb]
        embed_feat_ma = self.init_embed_mas(norm_mas)   # [B, n_mas, D_emb]
        embed_feat_veh = self.init_embed_vehs(norm_vehs)    # [B, n_vehs, D_emb]
        
        embed_feat_node = torch.cat([embed_feat_ope, embed_feat_ma, embed_feat_veh], dim=1)
        
        return embed_feat_node, proc_time, trans_time, offload_trans_time
    
    
    def get_normalized(self, 
            raw_opes, raw_mas, raw_vehs
        ):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param raw_vehs:
        
        :return: Normalized feats, including operations, machines and edges
        '''
    
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
        mean_vehs = torch.mean(raw_vehs, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_veh]
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
        std_vehs = torch.std(raw_vehs, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_veh]
        
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5), \
            (raw_vehs - mean_vehs) / (std_vehs + 1e-5))
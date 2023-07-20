import torch
import torch.nn as nn
from copy import deepcopy

from env.common_func import generate_trans_mat

class DTrans_embedder(nn.Module):
    def __init__(
            self,
            embedding_dim_,
            ope_feat_dim,
            ma_feat_dim,
            veh_feat_dim,
            **model_paras,
        ):
        super().__init__()
        self.job_centric = model_paras['job_centric']
        
        self.init_embed_opes = nn.Linear(ope_feat_dim, embedding_dim_)
        self.init_embed_mas = nn.Linear(ma_feat_dim, embedding_dim_)
        self.init_embed_vehs = nn.Linear(veh_feat_dim, embedding_dim_)
    
    def embedding(self, state,  encoder_version=0):
        '''
        :return ope_emb: [B, n_opes(n_jobs), D_emb]: 
        :return ma_emb: [B, n_mas, D_emb]
        :return veh_emb: [B, n_vehs, D_emb]
        :return porc_time: [B, n_opes(n_jobs), n_mas]
        :return onload_trans_time: [B, n_mas, n_mas]
        :return offload_trans_time: [B, n_vehs, n_mas]
        :return offload_trans_time_OV: [B, n_vehs, n_jobs]
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
        onload_trans_time_out = deepcopy(state.trans_times_batch[batch_idxes])   # [B, n_mas, n_mas]
        # [B, n_vehs, n_mas], [B, n_vehs, n_jobs]
        offload_trans_time_out, offload_trans_time_OV_out = self._get_offload_trans_time(state, onload_trans_time_out) 
        
        # === normalize ===
        norms = self.get_normalized(
            raw_opes, raw_mas, raw_vehs, proc_time, onload_trans_time_out, 
            offload_trans_time_out, offload_trans_time_OV_out
        )
        norm_opes = norms[0]
        norm_mas = norms[1]
        norm_vehs = norms[2]
        norm_proc_time = norms[3]
        norm_onload_trans_time = norms[4]
        norm_offload_trans_time = norms[5]
        norm_offload_trans_time_OV = norms[6]
        
        # === job centric ===
        if self.job_centric:
            norm_opes = norm_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, norm_opes.size(2)))
            proc_time_out = proc_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, proc_time.size(2)))
            # === normalize edge ===
            proc_time_out = norm_proc_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, norm_proc_time.size(2)))
            onload_trans_time_out = norm_onload_trans_time
            offload_trans_time_out = norm_offload_trans_time
            offload_trans_time_OV_out = norm_offload_trans_time_OV
        
        # === embedding opes, mas, vehs ===
        embed_opes = self.init_embed_opes(norm_opes)
        embed_mas = self.init_embed_mas(norm_mas)
        embed_vehs = self.init_embed_vehs(norm_vehs)
        
        return (embed_opes, embed_mas, embed_vehs,
                proc_time_out, onload_trans_time_out, offload_trans_time_out, offload_trans_time_OV_out)
        
        
        
    
    def _get_offload_trans_time(self, state, onload_trans_time):
        '''
        :param onload_trans_time: [B, n_mas, n_mas]
        
        :return offload_trans_time: [B, n_vehs, n_mas]
        :return offload_trans_time_OV: [B, n_vehs, n_jobs]
        '''
        batch_size, num_vehs = state.mask_veh_procing_batch.size()
        num_mas = state.mask_ma_procing_batch.size(1)
        veh_loc_batch = state.veh_loc_batch # [B, n_vehs]
        prev_ope_locs_batch = state.prev_ope_locs_batch # [B, n_jobs]
        allo_ma_batch = state.allo_ma_batch # [B, n_opes]
        elig_vehs = ~state.mask_veh_procing_batch   # [B, n_vehs]
        
        offload_trans_time = onload_trans_time.gather(
            1, veh_loc_batch[:, :, None].expand(-1, -1, onload_trans_time.size(2))
            )    # [B, n_vehs, n_mas]
        
        offload_trans_time_OV = offload_trans_time.gather(2, prev_ope_locs_batch[:, None, :].expand(-1, offload_trans_time.size(1), -1))
        # [B, n_vehs, n_jobs]
        
        return offload_trans_time, offload_trans_time_OV

    def get_normalized(self, 
            raw_opes, raw_mas, raw_vehs, proc_time, 
            onload_trans_time, offload_trans_time, offload_trans_time_OV
        ):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param raw_vehs:
        
        :return: Normalized feats, including operations, machines and vehicles
        '''
    
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
        mean_vehs = torch.mean(raw_vehs, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_veh]
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
        std_vehs = torch.std(raw_vehs, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_veh]
        
        proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        onload_trans_time_norm = self.feature_normalize(onload_trans_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        offload_trans_time_norm = self.feature_normalize(offload_trans_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        offload_trans_time_OV_norm = self.feature_normalize(offload_trans_time_OV)  # shape: [len(batch_idxes), num_opes, num_mas]
    
        
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5), \
            (raw_vehs - mean_vehs) / (std_vehs + 1e-5), \
            proc_time_norm, onload_trans_time_norm, \
            offload_trans_time_norm, offload_trans_time_OV_norm)
        
        
    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))
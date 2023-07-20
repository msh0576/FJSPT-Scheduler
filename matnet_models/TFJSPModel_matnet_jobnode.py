import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple
import math
import numpy as np

from matnet_models.TFJSPModel_sub import TFJSP_Encoder, FFSP_Decoder, MLPCritic, FJSP_Decoder
from env.common_func import select_vehicle, get_normalized

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    # logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                # logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)
    



class TFJSPModel_matnet_jobnode(nn.Module):
    '''
    Instead of O-M pair, use J-M pair
    '''
    def __init__(self,
                 embedding_dim_,
                 hidden_dim,
                 ope_feat_dim,
                 ma_feat_dim,
                 veh_feat_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 **model_paras
                 ):
        
        super().__init__()
        
        
        #
        self.embedding_dim = embedding_dim_
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = "greedy"
        self.temp = 1.0
        self.normalization = normalization
        self.tanh_clipping = tanh_clipping
        self.ope_feat_dim = ope_feat_dim
        self.ma_feat_dim = ma_feat_dim
        self.veh_feat_dim = veh_feat_dim
        self.all_feat_dim = ope_feat_dim + ma_feat_dim + veh_feat_dim + 1  # where 1: process time feature
        self.algo = None # matnet or heteronet
        self.model_paras = model_paras

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        
        # === NN models ===
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        self.init_embed_opes = nn.Linear(self.ope_feat_dim, self.embedding_dim)
        self.init_embed_mas = nn.Linear(self.ma_feat_dim, self.embedding_dim)
        self.init_embed_vehs = nn.Linear(self.veh_feat_dim, self.embedding_dim)
        self.init_embed_proc = nn.Linear(1, self.embedding_dim)
        self.init_embed_trans = nn.Linear(1, self.embedding_dim)
        
        self.init_embed = nn.Linear(self.all_feat_dim, self.embedding_dim)

        self.ffsp_encoder = TFJSP_Encoder(**model_paras)
        self.ffsp_decoder = FFSP_Decoder(**model_paras)
        
        assert self.embedding_dim % self.n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        
        
    def init(self, state):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        
        # === previous embeds, which is used for glipse ===
        self.prev_embed = torch.zeros(size=(self.batch_size, 1, self.embedding_dim))
    
    def act(self, state, baseline=False):
        return self.forward(state, baseline)
    
    def forward(self, state, baseline=False):
        '''
        Input
            input = state: namedtuple
        Output:
            action: [3, B]: (ope, ma, job)
            log_p_ope_ma: [B, n_opes + n_mas]
        '''
    
        batch_size = state.ope_ma_adj_batch.size(0)
        # === matnet encoding ===
        embed_feat_ope_ma_veh, embed_feat_job, embed_feat_ma,\
            embed_feat_veh, norm_proc_trans_time = self._init_embed(state)
        embedded_row, embedded_col = self.ffsp_encoder(embed_feat_job, embed_feat_ma, norm_proc_trans_time)    # [B, n_jobs, D_emb] | [B, n_mas, D_emb]
        embeddings = torch.cat([embedded_row, embedded_col], dim=1)
        
        
        self.ffsp_decoder.set_kv(embedded_row)
        # === encoded current machine ===
        # : pick one machine embeding
        encoded_current_ma, select_ma = self._get_encoding_matnet(embedded_col, state)  # [B, 1, D_emb] | [B, 1]

        # === get mask on selected machines ===
        # : repeat until selected machine has available opeartions
        ninf_mask = self._get_mask(state, select_ma)    # [B, n_jobs, n_mas] | [B, n_opes]
        ninf_mask_job = ninf_mask.gather(2, select_ma[:, None, :].expand(-1, ninf_mask.size(1), -1)).squeeze(2)    # [B, n_jobs]
        # === decoding ===
        all_jobs_prob = self.ffsp_decoder(encoded_current_ma, ninf_mask=ninf_mask_job.unsqueeze(1)).squeeze(1)    # [B, n_opes]

        batch_idxes = state.batch_idxes
        n = all_jobs_prob.size(1)
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        if self.training or self.model_paras['eval_type'] == 'softmax':
            while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                select_job = all_jobs_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1) # [B, 1]
                # shape: (batch, pomo)
                job_prob = all_jobs_prob.gather(1, select_job) # [B, 1]
                non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
                non_finish_batch[batch_idxes] = True
                finish_batch = torch.where(non_finish_batch == True, False, True)
                job_prob[finish_batch] = 1  # do not backprob finished episodes
                if (job_prob != 0).all():
                    break
        else:
            select_job = all_jobs_prob.argmax(dim=2)
            # shape: (batch, pomo)
            job_prob = torch.zeros(size=(batch_size, n))  # any number is okay
        select_ope = ope_step_batch.gather(1, select_job)   # [B, 1]
        
        # === select vehicle ===
        select_veh_dict = select_vehicle(state, select_ma, select_job)
        select_veh = select_veh_dict['veh_id'].unsqueeze(1).long() # [B, 1]
        
        # === make action ===
        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)  # [4, B]
        return action, job_prob
            
    
    def _get_encoding_matnet(self, encoded_nodes, state):
        '''
        Input:
            encoded_nodes: [B, n_mas, D_emb]
        Output:
            encoded_current_ma: [B, 1, D_emb]
            select_ma: [B, 1]
        '''
        batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        embed_dim = encoded_nodes.size(-1)
        
        _, ma_elig_on_ope = self._get_mask_ope_ma(state)    # [B, n_opes, n_mas]
        # : extract eligible machines that have at leat one available operation
        # print(f'ma_elig_on_ope:{ma_elig_on_ope.transpose(1,2)}')
        ma_elig_on_ope = ma_elig_on_ope.transpose(1, 2).any(dim=2)   # [B, n_mas]
        ma_elig_on_ope_float = torch.where(ma_elig_on_ope == True, 1.0, float('-inf'))
        
        select_ma = torch.softmax(ma_elig_on_ope_float, dim=1).multinomial(1)   # [B, 1]
        # === operation embeding on the selected machine ===
        encoded_current_ma = encoded_nodes.gather(1, select_ma[..., None].expand(-1, -1 , embed_dim))  # [B, 1, D_emb]
        
        return encoded_current_ma, select_ma
    
        
    
    def _init_embed(self, state):
        '''
        Output:
            embed_feat_job_ma_veh: [B, n_jobs + n_mas + n_vehs, D_emb]
            embed_feat_job: [B, n_jobs, D_emb]
            embed_feat_ma:  [B, n_mas, D_emb]
            embed_feat_jobs: [B, n_vehs, D_emb]
            norm_proc_trans_time: [B, n_jobs, n_mas]
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
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]   # [B, n_opes, ope_feat_dim]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes] # [B, n_mas, ma_feat_dim]
        raw_vehs = state.feat_vehs_batch.transpose(1, 2)[batch_idxes]   # [B, n_vehs, veh_feat_dim]
        proc_time = state.proc_times_batch[batch_idxes] # [B, n_opes, n_mas]
        trans_time = state.trans_times_batch[batch_idxes]   # [B, n_mas, n_mas]
        raw_jobs = raw_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, raw_opes.size(2)))    # [B, n_jobs, ope_feat_dim]
        proc_time_on_job = proc_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, proc_time.size(2)))    # [B, n_jobs, n_mas]
        
        
        # === edge cost matrix ===
        proc_trans_time = proc_time_on_job   # [B, n_jobs, n_mas]
        
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = get_normalized(raw_jobs, raw_mas, raw_vehs, proc_trans_time, trans_time, \
            flag_sample=True, flag_train=True)
        norm_jobs = (deepcopy(features[0]))
        norm_mas = (deepcopy(features[1]))
        norm_vehs = (deepcopy(features[2]))
        norm_proc_trans_time = (deepcopy(features[3]))
            
        
        # === embeds wrt n_jobs, n_mas and n_vehs ===
        embed_feat_job = self.init_embed_opes(norm_jobs)    # [B, n_jobs, D_emb]
        embed_feat_ma = self.init_embed_mas(norm_mas)   # [B, n_mas, D_emb]
        embed_feat_veh = self.init_embed_vehs(norm_vehs)    # [B, n_vehs, D_emb]
        embed_feat_proc = self.init_embed_proc(norm_proc_trans_time[..., None])    # [B, n_jobs, n_mas, D_emb]
        # embed_feat_trans = self.init_embed_trans(norm_trans_time_OM_pair[..., None]) # [B, n_mas, n_mas]
        embed_feat_job_ma_veh = torch.cat([embed_feat_job, embed_feat_ma, embed_feat_veh], dim=1)   # [B, n_jobs + n_mas + n_vehs, D_emb]
        
        return embed_feat_job_ma_veh, embed_feat_job, embed_feat_ma, embed_feat_veh, \
            norm_proc_trans_time
        
    
    def from_ope_to_job(self, select_ope, state):
        '''
        Input:
            select_ope: [B,]: selected operation index
        Output:
            selecct_job: [B,]
        '''
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        select_job = torch.where(ope_step_batch == select_ope[:, None].expand(-1, self.num_jobs))[1] # [B,]
        return select_job
    
    def random_act(self, state):
        '''
        Output:
            action: [ope_idx, mas_idx, job_idx]: [3,]
        '''
        batch_idxes = state.batch_idxes
        batch_size = len(batch_idxes)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_jobs = state.ope_step_batch.size(1)
        
        rand_act = torch.ones(size=(batch_size, num_mas * num_jobs,), dtype=torch.float)

        # === eligible action check: Detect eligible O-M pairs (eligible action) ===
        # = find some eligible operation indexes, which are non-finished ones =
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        
        
        
        dummy_shape = torch.zeros(size=(batch_size, num_jobs, num_mas))

        # = whether processing is possible. for each operation, there are processible machines =
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])   # [B, n_jobs, n_mas]
        
        # = whether machine/job/vehicle is possible =
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(dummy_shape) # [B, n_jobs, n_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(dummy_shape)   # [B, n_jobs, n_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)    # [B, n_jobs, n_mas]
        
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        
        # === get action_idx with masking the ineligible actions ===
        mask = eligible.transpose(1, 2).flatten(1)  # [B, n_mas * n_jobs]
        rand_act[~mask] = float('-inf')
        rand_act_prob = F.softmax(rand_act, dim=1)
        dist = Categorical(rand_act_prob)
        elig_act_idx = dist.sample()
        # ===== transper action_idx to indexes of machine, job and operation =====
        ma = (elig_act_idx / state.mask_job_finish_batch.size(1)).long()
        job = (elig_act_idx % state.mask_job_finish_batch.size(1)).long()
        ope = ope_step_batch[state.batch_idxes, job]

        return torch.stack((ope, ma, job), dim=0)
    
    def _get_mask(self, state, select_ma):
        '''
        Input:
            select_ma: [B, 1]
        Output:
            ninf_mask: [B, n_jobs, n_mas]
        '''
        batch_idxes = state.batch_idxes
        num_opes = state.ope_ma_adj_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        # === have processiable jobs on current selected machine ===
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])    # [B, n_jobs, n_mas]
        dummy_shape = torch.zeros(size=(len(batch_idxes), self.num_jobs, self.num_mas))
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(dummy_shape) # [B, n_jobs, n_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(dummy_shape)   # [B, n_jobs, n_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        mask = eligible  # [B, n_jobs, n_mas]
        ninf_mask = torch.where(mask == True, 0.0, -math.inf)
        # : get mask_job on a selected machine
        mask_job_on_select_ma = mask.gather(2, select_ma[:, None, :].expand(-1, self.num_jobs, -1)).squeeze(-1) # [B, n_jobs]
        if (~(mask_job_on_select_ma)).all():
            print("No eligible jobs on selected machine!")
            return
        
        
        return ninf_mask

        
    def _get_mask_ope_ma(self, state):
        '''
        Output:
            mask: [B, n_jobs, n_mas]
            mask_ope_ma: [B, n_opes, n_mas]
        '''
        batch_idxes = state.batch_idxes
        num_opes = state.ope_ma_adj_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_jobs = state.mask_job_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        opes_appertain_batch = state.opes_appertain_batch   # [B, n_opes]
        # machine mask
        mask_ma = ~state.mask_ma_procing_batch[batch_idxes] # [B, n_mas]
        
        # machine mask for each job
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])    # [B, n_jobs, n_mas]
        dummy_shape = torch.zeros(size=(len(batch_idxes), self.num_jobs, self.num_mas))
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(dummy_shape) # [B, n_jobs, n_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(dummy_shape)   # [B, n_jobs, n_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)

        if (~(eligible)).all():
            print("No eligible J-M pair!")
            return
        mask = eligible  # [B, n_jobs, n_mas]
        
        # === operation mask ===
        # : masks current ordered operation for each job
        mask_ope_step = torch.full(size=(self.batch_size, num_opes), dtype=torch.bool, fill_value=False) 
        tmp_batch_idxes = batch_idxes.unsqueeze(-1).repeat(1, num_jobs) # [B, n_jobs]
        mask_ope_step[tmp_batch_idxes, ope_step_batch] = True
        
        # : mask jobs that have no available machine and are processing
        mask_job = torch.where(mask.sum(dim=-1) > torch.zeros(size=(self.batch_size, self.num_jobs)), True, False)  # [B, n_jobs]
        mask_ope_by_job = mask_job.gather(1, opes_appertain_batch)
        
        mask_ope = mask_ope_by_job & mask_ope_step  # [B, n_opes]
        
        # === operation-machine mask ===
        mask_ope_padd = mask_ope[:, :, None].expand(-1, -1, num_mas)    # [B, n_opes, n_mas]
        mask_ma_padd = mask_ma[:, None, :].expand(-1, num_opes, -1) # [B, n_opes, n_mas]
        ope_ma_adj = state.ope_ma_adj_batch[batch_idxes]
        mask_ope_ma = mask_ope_padd & mask_ma_padd & (ope_ma_adj==1)  # [B, n_opes, n_mas]
        
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        
        return  mask, mask_ope_ma
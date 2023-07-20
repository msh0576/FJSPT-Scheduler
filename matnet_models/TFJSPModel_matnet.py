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
from env.common_func import select_vehicle, get_normalized, generate_trans_mat, select_vehicle_v2

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
    



class TFJSPModel_matnet(nn.Module):
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
        
        self.job_centric = model_paras['job_centric']
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
        
        
        
    def init(self, state, dataset=None, loader=None):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        
        # === previous embeds, which is used for glipse ===
        self.prev_embed = torch.zeros(size=(self.batch_size, 1, self.embedding_dim))
        
        embed_feat_ope_ma_veh, embed_feat_ope, embed_feat_ma,\
            embed_feat_veh, norm_proc_trans_time = self._init_embed(state)
        self.embedded_row, self.embedded_col = self.ffsp_encoder(embed_feat_ope, embed_feat_ma, norm_proc_trans_time)    # [B, n_opes, D_emb] | [B, n_mas, D_emb]
        self.embeddings = torch.cat([self.embedded_row, self.embedded_col], dim=1)
        
        self.ffsp_decoder.set_kv(self.embedded_row)
    
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
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        
        # === matnet encoding ===
        embed_feat_ope_ma_veh, embed_feat_ope, embed_feat_ma,\
            embed_feat_veh, norm_proc_trans_time = self._init_embed(state)
        embedded_row, embedded_col = self.ffsp_encoder(embed_feat_ope, embed_feat_ma, norm_proc_trans_time)    # [B, n_opes, D_emb] | [B, n_mas, D_emb]
        embeddings = torch.cat([embedded_row, embedded_col], dim=1)
        
        self.ffsp_decoder.set_kv(embedded_row)
        # === encoded current machine ===
        # : pick one machine embeding
        encoded_current_ma, select_ma = self._get_encoding_matnet(self.embedded_col, state)  # [B, 1, D_emb] | [B, 1]

        # === get mask on selected machines ===
        # : repeat until selected machine has available opeartions
        ninf_mask_ope_on_select_ma, ninf_mask\
            = self._get_mask(state, select_ma)    # [B, n_opes] | [B, n_jobs, n_mas]

        if self.job_centric:
            ninf_mask_on_select_ma = \
                ninf_mask.gather(2, select_ma[:, None, :].expand(-1, ninf_mask.size(1), -1)).squeeze(-1)    # [B, n_jobs]
        else:
            ninf_mask_on_select_ma = ninf_mask_ope_on_select_ma
            

        # === decoding ===
        all_opes_prob = self.ffsp_decoder(encoded_current_ma, ninf_mask=ninf_mask_on_select_ma.unsqueeze(1)).squeeze(1)    # [B, n_opes (n_jobs)]
        
        batch_idxes = state.batch_idxes
        n = all_opes_prob.size(1)
        if self.training or self.model_paras['eval_type'] == 'softmax':
            while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                select_job_ope = all_opes_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1) # [B, 1]
                # shape: (batch, pomo)
                job_ope_prob = all_opes_prob.gather(1, select_job_ope) # [B, 1]
                non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
                non_finish_batch[batch_idxes] = True
                finish_batch = torch.where(non_finish_batch == True, False, True)
                job_ope_prob[finish_batch] = 1  # do not backprob finished episodes
                if (job_ope_prob != 0).all():
                    break
        else:
            select_job_ope = all_opes_prob.argmax(dim=-1).unsqueeze(-1)
            # shape: (batch, pomo)
            job_ope_prob = torch.zeros(size=(batch_size, n))  # any number is okay
        
        if self.job_centric:
            select_job = select_job_ope
            select_ope = ope_step_batch.gather(1, select_job)   # [B, 1]
        else:
            select_ope = select_job_ope
            select_job = self.from_ope_to_job(select_ope.squeeze(1), state).unsqueeze(1)    # [B, 1]
        # === select vehicle ===
        # select_veh_dict = select_vehicle(state, select_ma, select_job)
        select_veh_dict = select_vehicle_v2(state, select_ma, select_job_ope, self.job_centric)
        select_veh = select_veh_dict['veh_id'].unsqueeze(1).long() # [B, 1]
        # === make action ===
        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)  # [4, B]
        return action, job_ope_prob
            
   
    
    
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
        
        # === select eligible machine ===
        batch_idxes = state.batch_idxes
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        ope_ma_adj = state.ope_ma_adj_batch[batch_idxes].bool() # [B, n_opes, n_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes] # [B, n_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] + state.mask_job_finish_batch[batch_idxes])   # [B, n_jobs]
        # : masks current ordered operation for each job
        mask_ope_step = torch.full(size=(len(batch_idxes), num_opes), dtype=torch.bool, fill_value=False)   
        for batch in batch_idxes:
            mask_ope_step[batch, ope_step_batch[batch]] = True  # [B, n_opes]
        # : expand eligible jobs into operation dimension
        mask_ope_by_job = torch.full(size=(batch_size, num_opes), dtype=torch.bool, fill_value=False)  # [B, n_opes]
        for batch in batch_idxes:
            mask_ope_by_job[batch] = torch.repeat_interleave(job_eligible[batch], state.nums_ope_batch[batch])
        # : set True into operation indexes that are processable and have avilable machines
        ope_eligible = mask_ope_by_job & mask_ope_step  # [B, n_opes]
        ope_eligible_add_ma = ope_eligible[:, :, None].expand(-1, -1, num_mas) & ma_eligible[:, None, :].expand(-1, num_opes, -1)   # [B, n_opes, n_mas]
        # : check eligible machines on eligible operations
        ma_elig_on_ope = ope_eligible_add_ma & ope_ma_adj  # [B, n_opes, n_mas]
        # : extract eligible machines that have at leat one available operation
        # print(f'ma_elig_on_ope:{ma_elig_on_ope.transpose(1,2)}')
        ma_elig_on_ope = ma_elig_on_ope.transpose(1, 2).any(dim=2)   # [B, n_mas]
        ma_elig_on_ope_float = torch.where(ma_elig_on_ope == True, 1.0, float('-inf'))
        
        select_ma = torch.softmax(ma_elig_on_ope_float, dim=1).multinomial(1)   # [B, 1]
        # === operation embeding on the selected machine ===
        encoded_current_ma = encoded_nodes.gather(1, select_ma[..., None].expand(-1, -1 , embed_dim))  # [B, 1, D_emb]
        
        return encoded_current_ma, select_ma
    
        
    
    def _init_embed(self, state, job_centric=True):
        '''
        Output:
            embed_feat_ope_ma: [B, n_opes + n_mas, D_emb]
            embed_feat_job_ma: [B, n_jobs + n_mas, D_emb]
            embed_feat_ope: [B, n_opes, D_emb]
            embed_feat_ma:  [B, n_mas, D_emb]
            norm_proc_trans_time: [B, n_opes, n_mas]
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
        
        
        # === edge cost matrix ===
        proc_trans_time = proc_time   # [B, n_opes, n_mas]
        
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = get_normalized(raw_opes, raw_mas, raw_vehs, proc_trans_time, trans_time, \
            flag_sample=True, flag_train=True)
        norm_opes = (deepcopy(features[0]))
        norm_mas = (deepcopy(features[1]))
        norm_vehs = (deepcopy(features[2]))
        norm_proc_trans_time = (deepcopy(features[3]))
            
        
        # === embeds wrt n_opes, n_mas and n_vehs ===
        if job_centric:
            norm_opes = norm_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, norm_opes.size(2)))
            norm_proc_trans_time = proc_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, norm_proc_trans_time.size(2)))
            
        
        
        embed_feat_ope = self.init_embed_opes(norm_opes)    # [B, n_opes, D_emb]
        embed_feat_ma = self.init_embed_mas(norm_mas)   # [B, n_mas, D_emb]
        embed_feat_veh = self.init_embed_vehs(norm_vehs)    # [B, n_vehs, D_emb]
        embed_feat_proc = self.init_embed_proc(norm_proc_trans_time[..., None])    # [B, n_opes, n_mas, D_emb]
        # embed_feat_trans = self.init_embed_trans(norm_trans_time_OM_pair[..., None]) # [B, n_mas, n_mas]
        embed_feat_ope_ma_veh = torch.cat([embed_feat_ope, embed_feat_ma, embed_feat_veh], dim=1)   # [B, n_opes + n_mas + n_vehs, D_emb]
        
        return embed_feat_ope_ma_veh, embed_feat_ope, embed_feat_ma, embed_feat_veh, \
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
            ninf_mask_ope_on_select_ma: [B, n_opes]: processible ope has 0.0, non-processible ope has -inf
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
        
        # === get mask_ope on a selected machine ===
        # : expand the mask_job_on_select_ma to the operation domain
        # : which consider job_eligible, ma_eligible
        mask_ope_on_select_ma = torch.full(size=(len(batch_idxes), num_opes), dtype=torch.bool, fill_value=False)
        for batch in batch_idxes:
            select_opes_on_select_ma = state.ope_step_batch[batch, mask_job_on_select_ma[batch]]    # consider job-eligible operations on a selected machine
            mask_ope_on_select_ma[batch, select_opes_on_select_ma] = True
        
        # : there exist one operation for each batch
        # if mask_ope_on_select_ma.any(dim=1).all() == False:
        if (~(mask_ope_on_select_ma)).all():
            print("No eligible opes on selected machine!")
            return
        ninf_mask_ope_on_select_ma = torch.where(mask_ope_on_select_ma == True, 0.0, float('-inf'))

        
        return ninf_mask_ope_on_select_ma, ninf_mask

        

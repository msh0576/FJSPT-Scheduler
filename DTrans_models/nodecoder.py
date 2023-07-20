import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical

from env.common_func import reshape_by_heads

from DHJS_models.decoder import TFJSP_Decoder_DHJS_Base


class TFJSP_NoDecoder_DTrans(TFJSP_Decoder_DHJS_Base):
    '''
    No context node, no edge embedding
    '''
    def __init__(self, **model_params):
        super().__init__(**model_params)
        
    def forward(self, embed_nodes, embed_context, prev_emb, state, mask, mask_ope_ma, mask_veh,
        training=False, eval_type='softmax', baseline=False,
        job_embedding=False):
        '''
        operation node selects, and then machine node
        Input:
            embed_nodes: [B, n_opes + n_mas + n_vehs, E]
            state:
            mask: [B, n_jobs, n_mas]
            mask_ope_ma: [B, n_opes, n_mas]
            
        '''
        head_num = self.model_params['head_num']
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        batch_idxes = state.batch_idxes
        
        
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
        non_finish_batch[batch_idxes] = True
        finish_batch = torch.where(non_finish_batch == True, False, True)
        

        if (training or eval_type == 'softmax') and (baseline == False):
            softmax = True
        else:
            softmax = False
        
        # === mask -> ninf_mask ===
        ninf_mask = torch.where(mask == True, 0., -math.inf)  # [B, n_jobs, n_mas]
        mask_ma = torch.where(mask.sum(dim=1) > 0, True, False)    # [B, n_mas]
        ninf_mask_ma = torch.where(mask_ma == True, 0., -math.inf)[:, None, :]    # [B, 1, n_mas]
        ninf_mask_veh = torch.where(mask_veh == True, 0., -math.inf)[:, None, :]    # [B, 1, n_vehs]
        
        if job_embedding:
            num_opes_jobs = num_jobs
            
            ninf_mask_ope_ma = torch.where(mask == True, 0., -math.inf)  # [B, n_jobs, n_mas]
            mask_ope = torch.where(mask.sum(dim=2) > 0, True, False)    # [B, n_jobs]
        else:
            num_opes_jobs = num_opes
            
            ninf_mask_ope_ma = torch.where(mask_ope_ma == True, 0., -math.inf)  # [B, n_opes, n_mas]
            mask_ope = torch.where(mask_ope_ma.sum(dim=2) > 0, True, False)    # [B, n_opes]
        ninf_mask_ope_job = torch.where(mask_ope == True, 0., -math.inf)[:, None, :]    # [B, 1, n_opes(n_jobs)]
        ninf_mask_nodes = torch.cat([ninf_mask_ope_job, ninf_mask_ma, ninf_mask_veh],dim=-1)    # [B, 1, n_nodes]
        
        
        # === graph embedding ===
        embed_graph = torch.mean(embed_nodes, dim=1, keepdim=True)  # [B, 1, D_emb]
        embed_opes = embed_nodes[:, :num_opes_jobs, :]
        embed_mas = embed_nodes[:, num_opes_jobs:num_opes_jobs+num_mas, :]
        embed_vehs = embed_nodes[:, num_opes_jobs+num_mas:, :]
        
        context = embed_graph
        
        # === select ope node ===
        q_ope = reshape_by_heads(self.Wq_ope(context), head_num=head_num)
        
        select_job_ope, ope_prob, _ = self._select_node(
            q_ope, self.k_ope, self.v_ope, ninf_mask_ope_job, 
            self.multi_head_combine_ope, self.single_head_key_ope,
            sqrt_embedding_dim, logit_clipping,
            batch_size, finish_batch, softmax
        )
        embed_select_ope = embed_opes.gather(1, select_job_ope[:, :, None].expand(-1, -1, embed_opes.size(-1)))
        
        if job_embedding:
            select_job = select_job_ope # [B, 1]
            select_ope = ope_step_batch.gather(1, select_job)   # [B, 1]
        else:
            select_ope = select_job_ope
            select_job = self.from_ope_to_job(select_ope.squeeze(1), ope_step_batch, num_jobs, num_opes)
        
        # === select ma node ===
        q_ma = reshape_by_heads(self.Wq_ma(context), head_num=head_num)
        ninf_mask_ma_on_ope = ninf_mask_ope_ma.gather(1, select_job_ope[:, :, None].expand(-1, -1, ninf_mask_ope_ma.size(2))) # [B, 1, n_mas]
        
        select_ma, ma_prob, _ = self._select_node(
            q_ma, self.k_ma, self.v_ma, ninf_mask_ma_on_ope, 
            self.multi_head_combine_ma, self.single_head_key_ma,
            sqrt_embedding_dim, logit_clipping,
            batch_size, finish_batch, softmax
        )
        
        embed_select_ma = embed_mas.gather(1, select_ma[:, :, None].expand(-1, -1, embed_mas.size(-1)))  # [B, 1, E]
        
        # === select veh node ===
        q_veh = reshape_by_heads(self.Wq_veh(context), head_num=head_num)
        select_veh, veh_prob, _ = self._select_node(
            q_veh, self.k_veh, self.v_veh, ninf_mask_veh, 
            self.multi_head_combine_veh, self.single_head_key_veh,
            sqrt_embedding_dim, logit_clipping,
            batch_size, finish_batch, softmax
        )
        embed_select_veh = embed_vehs.gather(1, select_veh[:, :, None].expand(-1, -1, embed_vehs.size(-1)))  # [B, 1, E]
        
        # === make action ===
        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)  # [4, B]
        prob = ope_prob * ma_prob * veh_prob
        
        select_embed = embed_select_ope + embed_select_ma + embed_select_veh # [B, 1, 3*D_emb]
        
        return action, prob.log(), select_embed.detach()

class TFJSP_DTrans_NonGraph(TFJSP_Decoder_DHJS_Base):
    '''
    No context node, no edge embedding
    '''
    def __init__(self, **model_params):
        super().__init__(**model_params)
        
    def forward(self, embed_nodes, embed_context, prev_emb, state, mask, mask_ope_ma, mask_veh,
        training=False, eval_type='softmax', baseline=False,
        job_embedding=False):
        '''
        operation node selects, and then machine node
        Input:
            embed_nodes: [B, n_opes + n_mas + n_vehs, E]
            state:
            mask: [B, n_jobs, n_mas]
            mask_ope_ma: [B, n_opes, n_mas]
            
        '''
        head_num = self.model_params['head_num']
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        batch_idxes = state.batch_idxes
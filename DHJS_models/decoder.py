import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical

from env.common_func import reshape_by_heads

class TFJSP_Decoder_DHJS_Base(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))
        
        self.proj_context = nn.Linear(4 * embedding_dim, embedding_dim, bias=True)

        self.Wq_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_ope = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_ma = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_veh = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # === for nodes key/value ===
        self.Wk_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # === for job key/value ===
        self.Wk_ope = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_ope = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # === for machine key/value ===
        self.Wk_ma = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_ma = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # === for vehicle key/value ===
        self.Wk_veh = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_veh = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        
        self.multi_head_combine_nodes = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_ope = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_ma = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_veh = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.proj_ope = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
    
    def set_nodes_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        
        self.k_nodes = reshape_by_heads(self.Wk_nodes(encoded_nodes), head_num=head_num)    # [B, H, n_opes, qkv_dim]
        self.v_nodes = reshape_by_heads(self.Wv_nodes(encoded_nodes), head_num=head_num)    # [B, H, n_opes, qkv_dim]
        
    def set_ope_kv(self, encoded_opes):
        '''
        Input:
            encoded_row: [B, n_opes, E]
        '''
        head_num = self.model_params['head_num']
        
        self.k_ope = reshape_by_heads(self.Wk_ope(encoded_opes), head_num=head_num)    # [B, H, n_opes, qkv_dim]
        self.v_ope = reshape_by_heads(self.Wv_ope(encoded_opes), head_num=head_num)    # [B, H, n_opes, qkv_dim]
        self.single_head_key_ope = encoded_opes.transpose(1, 2) # [B, E, n_opes+1]
        
    def set_ma_kv(self, encoded_mas):
        '''
        Input:
            encoded_mas: [B, n_mas, E]
        '''
        head_num = self.model_params['head_num']
        self.k_ma = reshape_by_heads(self.Wk_ma(encoded_mas), head_num=head_num)    # [B, H, n_mas, qkv_dim]
        self.v_ma = reshape_by_heads(self.Wv_ma(encoded_mas), head_num=head_num)    # [B, H, n_mas, qkv_dim]
        self.single_head_key_ma = encoded_mas.transpose(1, 2) # [B, E, n_mas]
    
    def set_veh_kv(self, encoded_vehs):
        '''
        Input:
            encoded_vehs: [B, n_vehs, E]
        '''
        head_num = self.model_params['head_num']
        self.k_veh = reshape_by_heads(self.Wk_veh(encoded_vehs), head_num=head_num)    # [B, H, n_vehs, qkv_dim]
        self.v_veh = reshape_by_heads(self.Wv_veh(encoded_vehs), head_num=head_num)    # [B, H, n_vehs, qkv_dim]
        self.single_head_key_veh = encoded_vehs.transpose(1, 2) # [B, E, n_vehs]
    
    
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
        
        
        
        
        # === get context embdding ===
        #: [opes, mas, vehs, graph, prev_emb]
        tmp_context = self.proj_context(torch.cat([embed_graph, prev_emb], dim=-1))  # [B, 1, D_emb]
        #: multi-head attention
        q_nodes = reshape_by_heads(self.Wq_nodes(tmp_context), head_num=head_num)   # [B, H, 1, qkv_dim]
        
        out_nodes = self._multi_head_attention_for_decoder(q_nodes, self.k_nodes, self.v_nodes,
                                                            rank3_ninf_mask=ninf_mask_nodes)  # [B, n (=1), H * qkv_dim]
        context = self.multi_head_combine_nodes(out_nodes)  # [B, 1, E]
        
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
        
        select_embed = torch.cat([embed_select_ope, embed_select_ma, embed_select_veh], dim=-1) # [B, 1, 3*D_emb]
        
        return action, prob.log(), select_embed.detach()
        
    def _select_node(self, q, k, v, ninf_mask, 
                    multi_head_combine, single_head_key,
                    sqrt_embedding_dim, logit_clipping,
                    batch_size, finish_batch,
                    softmax=True):
        '''
        Input:
            q:  [B, 1, E]
            k:  [B, n_node, E]
            v:  [B, n_node, E]
            ninf_mask: [B, 1, n_node]
            single_head_key: [B, E, n_node]
        Output:
            select_node: [B, 1]
            node_prob: [B, 1]
            mh_atten_out: [B, 1, E]
        '''
        out_concat = self._multi_head_attention_for_decoder(q, k, v,
                                                            rank3_ninf_mask=ninf_mask)  # [B, n (=1), H * qkv_dim]
        mh_atten_out = multi_head_combine(out_concat)  # [B, 1, E]

        # : Single-Head Attention, for probability calculation
        score = torch.matmul(mh_atten_out, single_head_key)    # [B, 1, n_opes or n_jobs]
        score_scaled = score / sqrt_embedding_dim

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask    # [B, 1, n_opes or n_jobs]
        all_node_prob = F.softmax(score_masked, dim=2).squeeze(1)   # [B, n_opes or n_jobs]
        # : select node 
        if softmax:
            dist = Categorical(all_node_prob)
            select_node = dist.sample().unsqueeze(-1)    # [B,1]
            node_prob = all_node_prob.gather(1, select_node) # [B, 1]
            node_prob[finish_batch] = 1  # do not backprob finished episodes
            # while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
            #     select_node = all_node_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1) # [B, 1]
            #     node_prob = all_node_prob.gather(1, select_node) # [B, 1]
            #     node_prob[finish_batch] = 1  # do not backprob finished episodes
            #     if (node_prob != 0).all():
            #         break
        else:
            select_node = all_node_prob.argmax(dim=1).unsqueeze(1)    # [B, 1]
            node_prob = torch.zeros(size=(batch_size, 1))  # any number is okay
        
        
        
        
        return select_node, node_prob, mh_atten_out
    
    def from_ope_to_job(self, select_ope, ope_step_batch, num_jobs, num_opes):
        '''
        Input:
            ope_step_batch: [B, n_jobs]
            select_ope: [B,]: selected operation index
        Output:
            selecct_job: [B, 1]
        '''
        batch_size = ope_step_batch.size(0)
        empty_ope_batch = torch.ones(size=(batch_size, 1)) * num_opes
        # ope_step_batch_plus1 = torch.cat([ope_step_batch, empty_ope_batch], dim=1) # [B, n_jobs + 1]
        # select_job = torch.where(ope_step_batch1 == select_ope[:, None].expand(-1, num_jobs+1))[1] # [B,]
        select_job = torch.where(ope_step_batch == select_ope[:, None].expand(-1, num_jobs))[1] # [B,]
        return select_job.unsqueeze(1)
        
        
    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        '''
        Input:
            q: [B, H, 1, qkv_dim]
            k, v: [B, H, n_opes, qkv_dim]
            rank2_ninf_mask: [B, n_opes]
            rank3_ninf_mask: [B, n_mas, n_opes] or [B, 1, n_opes]
        '''
        batch_size = q.size(0)
        n = q.size(2)
        num_node = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))  # [B, H, n, n_opes (or n_mas)]

        score_scaled = score / sqrt_qkv_dim

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_size, head_num, n, num_node)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, n, num_node)

        weights = nn.Softmax(dim=3)(score_scaled)   # [B, H, n, n_opes (or n_mas)]

        out = torch.matmul(weights, v)  # [B, H, n, qkv_dim]

        out_transposed = out.transpose(1, 2)    # [B, n, H, qkv_dim]

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)  # [B, n, H *qkv_dim]

        return out_concat


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from env.common_func import reshape_by_heads
from DHJS_models.decoder import TFJSP_Decoder_DHJS_Base
from DHJS_models.attention_networks import mlp_layer, query_att_batch, Attention_batch
from env.common_func import generate_trans_mat, select_vehicle

class TFJSP_Decoder_DHJS_V2(TFJSP_Decoder_DHJS_Base):
    def __init__(self, **model_params):
        super().__init__(**model_params)
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        self.proj_glimp_ope = nn.Linear(4 * embedding_dim, embedding_dim, bias=True)
        self.proj_glimp_ma = nn.Linear(4 * embedding_dim, embedding_dim, bias=True)
        self.proj_glimp_veh = nn.Linear(4 * embedding_dim, embedding_dim, bias=True)
        
        
        self.atten_to_job = query_att_batch(embedding_dim)
        self.atten_to_ma = query_att_batch(embedding_dim)
        self.atten_to_veh = query_att_batch(embedding_dim)
        
        self.job_atten = Attention_batch(embedding_dim, embedding_dim, False)
        self.ma_atten = Attention_batch(embedding_dim, embedding_dim, False)
        self.veh_atten = Attention_batch(embedding_dim, embedding_dim, False)
        
        self.global1_merge = mlp_layer(4 * embedding_dim, embedding_dim)
        self.global2_merge = mlp_layer(4 * embedding_dim, embedding_dim)
        self.global3_merge = mlp_layer(4 * embedding_dim, embedding_dim)
        
        self.proj_select_ope_ma = nn.Linear(2 * embedding_dim, embedding_dim, bias=True)
        self.proj_select_ope_ma_veh = nn.Linear(3 * embedding_dim, embedding_dim, bias=True)
        
    
    def _select_node(self, global_merge, ninf_mask, 
                    single_head_key, 
                    sqrt_embedding_dim, logit_clipping,
                    batch_size, finish_batch,
                    softmax=True):
        '''
        Input:
            global_merge: [B, 1, E]
            ninf_mask: [B, 1, n_node]
            single_head_key: [B, E, n_node]
        Output:
            select_node: [B, 1]
            node_prob: [B, 1]
        '''
        score = torch.matmul(global_merge, single_head_key)    # [B, 1, n_opes or n_jobs]
        score_scaled = score / sqrt_embedding_dim

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask    # [B, 1, n_opes or n_jobs]
        all_node_prob = F.softmax(score_masked, dim=2).squeeze(1)   # [B, n_opes or n_jobs]
        
        # : select node 
        if softmax:
            while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                select_node = all_node_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1) # [B, 1]
                node_prob = all_node_prob.gather(1, select_node) # [B, 1]
                node_prob[finish_batch] = 1  # do not backprob finished episodes
                if (node_prob != 0).all():
                    break
        else:
            select_node = all_node_prob.argmax(dim=1).unsqueeze(1)    # [B, 1]
            node_prob = torch.zeros(size=(batch_size, 1))  # any number is okay
        return select_node, node_prob

    def _get_global_atten(self, context, embed_opes, embed_mas, embed_vehs):
        '''
        Input:
            context: [B, 1, D_emb]
            embed_opes: [B, n_opes, D_emb]
            embed_mas: [B, n_mas, D_emb]
            embed_vehs: [B, n_vehs, D_emb]
        Output:
            global_merge: [B, 1, D_emb]
        '''
        global_ope = self.atten_to_job(context.squeeze(1), embed_opes)    # [B, 1, E]
        global_ma = self.atten_to_ma(context.squeeze(1), embed_mas)
        global_veh = self.atten_to_veh(context.squeeze(1), embed_vehs)
        
        global_node = torch.cat([context, global_ope, global_ma, global_veh], dim=-1)    # [B, 1, 4*E]
        global_merge = self.proj_glimp_ope(global_node) # [B, 1, E]
        
        return global_merge
        

    def forward(self, embed_nodes, embed_context, prev_emb, state, mask, mask_ope_ma, mask_veh,
                training=False, eval_type='softmax', baseline=False,
                job_embedding=False):
        '''
        operation node selects, and then machine node
        Input:
            embed_nodes: [B, n_opes + n_mas + n_vehs, E]
            embed_context: [B, 1, E]
            prev_emb: [B, 1, E]
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
        
        if job_embedding:
            num_opes_jobs = num_jobs
        else:
            num_opes_jobs = num_opes
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
        non_finish_batch[batch_idxes] = True
        finish_batch = torch.where(non_finish_batch == True, False, True)
        

        if (training or eval_type == 'softmax') and (baseline == False):
            softmax = True
        else:
            softmax = False
        
        # === node embedding ===
        # embed_opes = embed_nodes[:, :num_opes, :]
        embed_graph = embed_nodes.mean(dim=1, keepdim=True)
        embed_opes = embed_nodes[:, :num_opes_jobs, :]
        embed_mas = embed_nodes[:, num_opes_jobs:num_opes_jobs+num_mas, :]
        embed_vehs = embed_nodes[:, num_opes_jobs+num_mas:, :]
        
        # === mask -> ninf_mask ===
        ninf_mask = torch.where(mask == True, 0., -math.inf)  # [B, n_jobs, n_mas]
        mask_ma = torch.where(mask.sum(dim=1) > 0, True, False)    # [B, n_mas]
        ninf_mask_ma = torch.where(mask_ma == True, 0., -math.inf)[:, None, :]    # [B, 1, n_mas]
        ninf_mask_ope_ma = torch.where(mask_ope_ma == True, 0., -math.inf)  # [B, n_opes, n_mas]
        ninf_mask_veh = torch.where(mask_veh == True, 0., -math.inf)[:, None, :]    # [B, 1, n_vehs]
        mask_job = torch.where(mask.sum(dim=2) > 0, True, False)    # [B, n_jobs]
        mask_ope = torch.where(mask_ope_ma.sum(dim=2) > 0, True, False)    # [B, n_opes]
        ninf_mask_ope_job = torch.where(mask_ope == True, 0., -math.inf)[:, None, :]    # [B, 1, n_opes]
        
        
        # === select ope node ===
        global1_merge = self._get_global_atten(prev_emb, embed_opes, embed_mas, embed_vehs)
        
        select_ope, ope_job_prob = self._select_node(
            global1_merge, ninf_mask_ope_job, 
            self.single_head_key_ope, 
            sqrt_embedding_dim, logit_clipping,
            batch_size, finish_batch,
            softmax=softmax
        )
    
        embed_select_ope = embed_opes.gather(1, select_ope[:, :, None].expand(-1, -1, embed_opes.size(-1)))
        select_job = self.from_ope_to_job(select_ope.squeeze(1), ope_step_batch, num_jobs, num_opes)
        
        # === select ma node ===
        global2_merge = self._get_global_atten(embed_select_ope, embed_opes, embed_mas, embed_vehs)
        ninf_mask_ma_on_job = ninf_mask.gather(1, select_job[:, :, None].expand(-1, -1, ninf_mask.size(2))) # [B, 1, n_mas]
        
        select_ma, ma_prob = self._select_node(
            global2_merge, ninf_mask_ma_on_job, 
            self.single_head_key_ma, 
            sqrt_embedding_dim, logit_clipping,
            batch_size, finish_batch,
            softmax=softmax
        )
        embed_select_ma = embed_mas.gather(1, select_ma[:, :, None].expand(-1, -1, embed_mas.size(-1)))  # [B, 1, E]
        
        
        # === select vehicle node ===
        global3_merge = self._get_global_atten(embed_select_ope, embed_opes, embed_mas, embed_vehs)
        
        select_veh, veh_prob = self._select_node(
            global3_merge, ninf_mask_veh, 
            self.single_head_key_veh, 
            sqrt_embedding_dim, logit_clipping,
            batch_size, finish_batch,
            softmax=softmax
        )
        embed_select_veh = embed_vehs.gather(1, select_veh[:, :, None].expand(-1, -1, embed_vehs.size(-1)))  # [B, 1, E]
        embed_select_ope_ma_veh = self.proj_select_ope_ma_veh(torch.cat([embed_select_ope, embed_select_ma, embed_select_veh], dim=-1))
        
        
        # === make action ===
        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)  # [4, B]
        prob = ope_job_prob * ma_prob * veh_prob
        

        return action, prob.log(), embed_select_ope.detach()

class TFJSP_Decoder_DHJS_V3(TFJSP_Decoder_DHJS_Base):
    '''
    use nearest vehicle selection (NVS)
    '''
    def __init__(self, **model_params):
        super().__init__(**model_params)
        embedding_dim = model_params['embedding_dim']
        self.transtime_btw_ma_max = model_params['transtime_btw_ma_max']
        self.prev_sel_emb = torch.zeros(size=(model_params['batch_size'], 3, embedding_dim))
        
        self.proj_context = nn.Linear(4 * embedding_dim, embedding_dim, bias=True)
    
    def forward(self,
        embed_nodes, embed_context, prev_emb, state, mask, mask_ope_ma, mask_veh,
        training=False, eval_type='softmax', baseline=False,
        job_embedding=False
    ):
        '''
        Input:
            prev_emb: [B, 3, D_emb]
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
        
        # === select vehicle node ===
        select_veh_dict = select_vehicle(state, select_ma, select_job)
        select_veh = select_veh_dict['veh_id'].unsqueeze(1).long() # [B, 1]
        embed_select_veh = embed_vehs.gather(1, select_veh[:, :, None].expand(-1, -1, embed_vehs.size(-1)))  # [B, 1, E]
        
        
        # === make action ===
        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)  # [4, B]
        prob = ope_prob * ma_prob
        
        select_embed = torch.cat([embed_select_ope, embed_select_ma, embed_select_veh], dim=-1) # [B, 1, 3*D_emb]
        
        return action, prob.log(), select_embed.detach()

    
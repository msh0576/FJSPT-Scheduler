import torch
from torch import nn
import torch_geometric.nn as gnn
from sat_models.sat.layers import TransformerEncoderLayer
from einops import repeat
from DHJS_models.embedder import DHJS_embedder
from DHJS_models.decoder_types import TFJSP_Decoder_DHJS_V2, TFJSP_Decoder_DHJS_V3
from DHJS_models.decoder import TFJSP_Decoder_DHJS_Base
import torch_geometric.utils as utils


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
            ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index, 
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean',
                 ope_feat_dim=8, ma_feat_dim=4, veh_feat_dim=5,
                 encoder_version=1, decoder_version=1,
                 batch_size=128,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.device =kwargs['device']

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model) 
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))
        
        # === embedder ===
        self.embedder = DHJS_embedder(
            d_model, ope_feat_dim, ma_feat_dim, veh_feat_dim,
            **kwargs
        )
        
        # === decoder ===
        self.decoder_version = decoder_version
        self.prev_embed = torch.zeros(size=(batch_size, 1, d_model))
        if decoder_version == 1:
            decoder_fn = TFJSP_Decoder_DHJS_Base
            self.prev_embed = torch.zeros(size=(batch_size, 1, 3*d_model))
        elif decoder_version == 2:
            decoder_fn = TFJSP_Decoder_DHJS_V2
        elif decoder_version == 3:
            decoder_fn = TFJSP_Decoder_DHJS_V3
            self.prev_embed = torch.zeros(size=(batch_size, 1, 3*d_model))
        else:
            raise Exception('decoder version error!')
        self.decoder = decoder_fn(**kwargs)
        
        # ===
        self.job_embedding = False

    def init(self, state, dataset=None, loader=None):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)

        self.dataset = dataset
        
        self.edge_index_list = []
        self.edge_attr_list = []
        self.subgraph_node_index_list = []
        self.subgraph_edge_index_list = []
        self.subgraph_indicator_index_list = []
        self.complete_edge_index_list = []
        self.subgraph_edge_attr_list = []
        self.degree_list = []
        for i, data in enumerate(self.dataset):
            # print(f'data:{data}')
            data = data.to(self.device)
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            # edge_attr = edge_attr.to(self.device)
            node_depth = data.node_depth if hasattr(data, "node_depth") else None
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator 
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                                    else None
            

            complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
            abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
            degree = data.degree if hasattr(data, 'degree') else None
            # output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
    
            print(f'subgraph_edge_attr:{subgraph_edge_attr.shape}')
            subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
            print(f'--subgraph_edge_attr:{subgraph_edge_attr.shape}')
            
            self.edge_index_list.append(edge_index)
            self.edge_attr_list.append(edge_attr)
            self.subgraph_node_index_list.append(subgraph_node_index)
            self.subgraph_edge_index_list.append(subgraph_edge_index)
            self.subgraph_indicator_index_list.append(subgraph_indicator_index)
            self.complete_edge_index_list.append(complete_edge_index)
            self.subgraph_edge_attr_list.append(subgraph_edge_attr)
            self.degree_list.append(degree)
            
            
            
    def act(self, state, baseline=False):
        return self.forward(state, baseline)
    
    def forward(self, state, baseline=False, return_attn=False):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_jobs = state.mask_job_finish_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_vehs = state.mask_veh_procing_batch.size(1)
            
            
        # print(f'output:{output, output.shape}')   # [n_node, D_emb]
        # print(f'edge_index:{edge_index, edge_index.shape}')   # [2, n_edge]
        # print(f'subgraph_node_index:{subgraph_node_index, subgraph_node_index.shape}') # [sub_n_node]
        # print(f'subgraph_edge_index:{subgraph_edge_index, subgraph_edge_index.shape}') # [2, sub_n_edge]
        # print(f'subgraph_edge_attr:{subgraph_edge_attr, subgraph_edge_attr.shape}')   # [sub_n_edge, D_emb_edge]
        # print(f'subgraph_indicator_index:{subgraph_indicator_index, subgraph_indicator_index.shape}')   # [sub_n_node]
        
        embed_feat_ope, embed_feat_ma,\
            embed_feat_veh, norm_proc_trans_time, norm_offload_trans_time, \
            norm_trans_time, oper_adj_batch, _, norm_MVpair_trans_time, norm_onload_trans_time, \
            mask_dyn_ope_ma_adj, mask_ma \
            = self.embedder.embedding(state, encoder_version=10)
        embed = torch.cat([embed_feat_ope, embed_feat_ma, embed_feat_veh], dim=-2)[0]   # [n_node, D_emb]

        encod = []
        for batch in range(self.batch_size):
            output = self.encoder(
                embed, 
                self.edge_index_list[batch], 
                self.complete_edge_index_list[batch],
                edge_attr=self.edge_attr_list[batch], 
                degree=self.degree_list[batch],
                subgraph_node_index=self.subgraph_node_index_list[batch],
                subgraph_edge_index=self.subgraph_edge_index_list[batch],
                subgraph_indicator_index=self.subgraph_indicator_index_list[batch], 
                subgraph_edge_attr=self.subgraph_edge_attr_list[batch],
                # ptr=data.ptr,
                ptr=None,
                return_attn=False
            )   # [n_node, D_emb]
            encod.append(output)
        encod = torch.stack(encod)
        
        encod_ope = encod[:, :num_opes, :]
        encod_ma = encod[:, num_opes:num_opes+num_mas, :]
        encod_veh = encod[:, num_opes+num_mas:, :]
        
        # === decoding ===
        action, log_p = self._get_action_with_decoder(state, encod_ope, encod_ma, encod_veh, baseline=baseline)
        # print(f'action:{action} | log_p:{log_p}')
        
        # action = random_act(state)
        return action, log_p
    
    def _get_action_with_decoder(self, state, embedded_ope, embedded_ma, embedded_veh, baseline):
        '''
        Input:
            state:
            embedding: [B, n_nodes, D_emb]
        Output:
            action: [3, B]
            log_p: [B, 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        if self.job_embedding:
            num_opes_jobs = num_jobs
        else:
            num_opes_jobs = num_opes
        
        # === get mask ===
        mask, mask_ope_ma = self._get_mask_ope_ma(state) # [B, n_opes, n_mas]
        mask_veh = ~state.mask_veh_procing_batch   # [B, n_vehs]

        # === preprocess decoding ===
        embedding = torch.cat([embedded_ope, embedded_ma, embedded_veh], dim=1) # [B, n_nodes, D_emb]
        self.decoder.set_nodes_kv(embedding)
        self.decoder.set_ope_kv(embedded_ope)
        self.decoder.set_ma_kv(embedded_ma)
        self.decoder.set_veh_kv(embedded_veh)
        
        # === decoder ===
        action, log_p, prev_embed = self.decoder(
            embedding, None, self.prev_embed, state, mask, mask_ope_ma, mask_veh,
            training=self.training, eval_type=self.kwargs['eval_type'], baseline=baseline,
            job_embedding=self.job_embedding
        )  # [B, n_opes]
        self.prev_embed = prev_embed
        
        
        return action, log_p
        
    
    
    def _get_mask_ope_ma(self, state):
        '''
        Output:
            mask: [B, n_jobs, n_mas]
            mask_ope_ma: [B, n_opes, n_mas]
        '''
        batch_idxes = state.batch_idxes
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        opes_appertain_batch = state.opes_appertain_batch   # [B, n_opes]
        # machine mask
        mask_ma = ~state.mask_ma_procing_batch[batch_idxes] # [B, n_mas]
        
        # machine mask for each job
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])    # [B, n_jobs, n_mas]
        dummy_shape = torch.zeros(size=(len(batch_idxes), num_jobs, num_mas))
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
        mask_ope_step = torch.full(size=(batch_size, num_opes), dtype=torch.bool, fill_value=False) 
        tmp_batch_idxes = batch_idxes.unsqueeze(-1).repeat(1, num_jobs) # [B, n_jobs]
        mask_ope_step[tmp_batch_idxes, ope_step_batch] = True
        
        # : mask jobs that have no available machine and are processing
        mask_job = torch.where(mask.sum(dim=-1) > torch.zeros(size=(batch_size, num_jobs)), True, False)  # [B, n_jobs]
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
    
    def _make_heads(self, v, num_steps=None):
        '''
        Ex) v = glimpse_key_fixed [B, 1, n_opes + n_mas, D_emb] -> [B, 1, n_opes + n_mas, H, D_emb/H] -> [H, B, 1, n_opes + n_mas, D_emb/H]
        '''
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )       
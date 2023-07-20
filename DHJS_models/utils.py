import torch
import numpy as np
import pandas as pd
import networkx as nx


def get_core_adj_list(num_opes, num_mas, num_vehs, adj_mat, device, spmat_format=True):
    '''
    Input:
        adj_mat: [B, n_opes, n_opes+n_mas+n_vehs]
    Output:
        batch_core_adj_list: list of k_core_graph_list: len=batch_size
        k_core_graph_list: list of k-core graph: len=max_k_core on the batch instance
        pad_batch_core_adj_list: [B, max_core, n_nodes, n_nodes]
    '''
    batch_size = adj_mat.size(0)
    num_nodes = num_opes + num_mas + num_vehs
    batch_core_adj_list = []
    max_core = 0
    for batch in range(batch_size):
        # === get k-core graph list ===
        # list: [max_kcore, num_nodes, num_nodes]
        k_core_graph_list = get_kcore_graph(num_opes, num_mas, num_vehs, adj_mat[batch], spmat_format)
        k_core_graph_list = k_core_graph_list[::-1] # reverse order: max core, (max-1) core, ... 1 core
        if not spmat_format:
            k_core_graph_np = np.array(k_core_graph_list)
            k_core_graph_tensor = torch.from_numpy(k_core_graph_np).float().to(device)
        else:
            raise Exception('cannot handle spmat_format!')
        batch_core_adj_list.append(k_core_graph_tensor)
        core_size = k_core_graph_tensor.size(0)
        if max_core < core_size:
            max_core = core_size
    
    # === pad zero tensor ===
    pad_batch_core_adj_list = torch.zeros(size=(batch_size, max_core, num_nodes, num_nodes))
    for batch in range(batch_size):
        num_core = batch_core_adj_list[batch].size(0)
        pad_batch_core_adj_list[batch, :num_core, :, :] = batch_core_adj_list[batch]
    return pad_batch_core_adj_list

def get_kcore_graph(num_opes, num_mas, num_vehs, adj_mat, spmat=True, draw_fig=False):
    '''
    Input:
        adj_mat: [n_opes, n_opes + n_mas + n_vehs]
    Output:
        k_core_graph_list = [k_core_graph, ...] length: max_core_num 
    '''
    num_nodes = num_opes + num_mas + num_vehs
    
    full_nodes_list = np.arange(num_nodes)
    graph = _get_nx_graph(num_opes, num_mas, num_vehs, adj_mat)
    
    
    core_num_dict = nx.core_number(graph)
    max_core_num = max(list(core_num_dict.values()))
    
    k_core_graph_list = []
    for i in range(1, max_core_num+1):
        k_core_graph = nx.k_core(graph, k=i, core_number=core_num_dict)
        k_core_graph.add_nodes_from(full_nodes_list)
        if spmat:
            A = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=full_nodes_list)
        else:
            A = nx.to_numpy_array(k_core_graph, nodelist=full_nodes_list)
        k_core_graph_list.append(A)
    
    return k_core_graph_list

def _get_nx_graph(num_opes, num_mas, num_vehs, adj_mat):
    '''
    Input:
        adj_mat: [n_opes, n_opes + n_mas + n_vehs]
    '''
    num_nodes = num_opes + num_mas + num_vehs
    adj_pd = _generate_df_graph(adj_mat)
    # print(f'adj_pd;{adj_pd}')
    
    graph = nx.from_pandas_edgelist(adj_pd, "from_id", "to_id", edge_attr='weight', create_using=nx.Graph)
    graph.add_nodes_from(np.arange(num_nodes))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def _generate_df_graph(adj_mat):
    '''
    an operation is connected with its compatible machines
    Input:
        adj_mat: [n_opes, n_opes + n_mas + n_vehs]
    Output:
        adj_pd: pandas
            from_id | to_id | weight (1 if they are connected)
            
    '''
    adj_mat_np = adj_mat.detach().cpu().numpy()   # [num_opes, num_mas]
    
    adj_mat_edges = np.transpose(np.nonzero(adj_mat_np))
    adj_pd = pd.DataFrame(adj_mat_edges, columns=['from_id', 'to_id'])
    adj_pd['weight'] = 1
    return adj_pd
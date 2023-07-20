import torch
from copy import deepcopy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random

from DHJS_models.utils import _get_nx_graph
from env.case_generator_kcore import CaseGenerator_kcore
from env.case_generator_v2 import CaseGenerator
from env.tfjsp_env import TFJSPEnv

def subgraph_meta_learner(
    minibatch_size, num_opes, num_mas, num_vehs, 
    proctime_per_ope_max, transtime_btw_ma_max,
    env_paras, device, dynamic
):
    '''
    : return kcore_env_list:
    : return minibatch_size_list: 
    '''
    # every epoch, sample num_opes, num_mas, num_vehs
    tmp_num_opes = random.randint(2, num_opes)
    tmp_num_mas = random.randint(2, num_mas)
    tmp_num_vehs = random.randint(2, num_vehs)
    
    # generate main graph 
    main_graph = _generate_env(
        env_paras, tmp_num_opes, tmp_num_mas, tmp_num_vehs,
        device, dynamic, proctime_per_ope_max, transtime_btw_ma_max
    )
    
    
    ope_ope_adj = main_graph.ope_adj_batch.float()   # [B, n_opes, n_opes]
    ope_ma_adj = torch.where(main_graph.ope_ma_adj_batch>0, 1., 0.) # [B, n_opes, n_mas]
    ope_veh_adj = torch.where(main_graph.ope_veh_adj_batch>0, 1., 0.)           # [B, n_opes, n_vehs]
    adj_mat_batch = torch.cat([ope_ope_adj, ope_ma_adj], dim=-1) # [B, n_opes, n_opes + n_mas]
    
    # sample one batch
    batch_size = adj_mat_batch.size(0)
    rand_batch_idx = random.randint(0, batch_size-1)
    
    # get kcore environment cost (subgraphs) given random instance
    kcore_case_list = get_kcore_case_list_from_adj_mat(
        tmp_num_opes, tmp_num_mas, tmp_num_vehs, adj_mat_batch[rand_batch_idx], device,
        proctime_per_ope_max, transtime_btw_ma_max
    )
    
    # given case_list, generate env list
    num_kcore = len(kcore_case_list)
    kcore_env_list = []
    minibatch_size_list = [minibatch_size] * num_kcore
    assert env_paras['meta_rl']['enable'] == True
    
    for idx, case in enumerate(kcore_case_list):
        env_paras['meta_rl']['minibatch'] = minibatch_size_list[idx]
        env = TFJSPEnv(case=case, env_paras=env_paras)
        kcore_env_list.append(env)
    return kcore_env_list, minibatch_size_list
    
    
    
    
    
def get_kcore_case_list_from_adj_mat(
    num_opes, num_mas, num_vehs, adj_mat, device,
    proctime_per_ope_max, transtime_btw_ma_max
):
    '''
    : param adj_mat: [num_opes, num_opes + num_mas]
    : return kcore_case_list: list of kcore case
    '''
    # === generate kcore sub-graphs given an instance ===
    k_core_graph_list, k_core_conn_opes, k_core_num_opes = get_kcore_graph(num_opes, num_mas, num_vehs, adj_mat, spmat=False, draw_fig=True)
    k_core_graph_list = k_core_graph_list[::-1] # reverse order: max core, (max-1) core, ... 1 core
    k_core_conn_opes = k_core_conn_opes[::-1]
    k_core_num_opes = k_core_num_opes[::-1]
    
    for idx, kcore_graph in enumerate(k_core_graph_list):
        k_core_graph_list[idx] = torch.from_numpy(kcore_graph).float().to(device)
    # k_core_graph_np = np.array(k_core_graph_list)
    # k_core_graph_tensor = torch.from_numpy(k_core_graph_np).float().to(device)
    
    # === get feasible subgraphs given kcore subgraphs ===
    kcore_redu_ope_ma_adj, kcore_redu_conn_opes, kcore_redu_num_opes = \
        get_feasible_subgraphs(k_core_graph_list, k_core_conn_opes, k_core_num_opes, num_opes, num_mas)
    
    # === case generator ===
    kcore_case_list = []
    for idx, kcore_graph in enumerate(k_core_graph_list):
        case = CaseGenerator_kcore(
            sum(kcore_redu_num_opes[idx]), num_mas, num_vehs, device,
            proctime_per_ope_max, transtime_btw_ma_max,
            kcore_redu_ope_ma_adj=kcore_redu_ope_ma_adj[idx],
            kcore_redu_num_opes=kcore_redu_num_opes[idx]
        )
        kcore_case_list.append(case)
    
    return kcore_case_list
    
    
    
    
def get_feasible_subgraphs(
    kcore_graph_list, kcore_conn_opes, kcore_num_opes,
    num_opes, num_mas
    ):
    '''
    : param kcore_graph_list: 
        [max-core graph, max-1 core graph, ...]
    : param kcore_conn_opes: 
        [{job_0:[num_opes_list], job_1: [], ...}, {}, {}, ...]
    : param kcore_num_opes: list of kcore graph num_opes
        [[k-th core graph's num_opes], [], ...]
    : param num_opes:
    : param num_mas:
    : return kcore_redu_ope_ma_adj: list of kcore ope-ma adjacent matrix: size of each matrix [elig_num_opes, num_mas]
    : return kcore_redu_conn_opes: 
    : return kcore_redu_num_opes: remove non-eligible nodes from kcore_num_opes
    '''
    # === reduce sub-graph by removing the non-eligible operations ===
    num_kcore_graph = len(kcore_graph_list)
    non_elig_nodes = []
    kcore_redu_ope_ma_adj = []
    kcore_redu_conn_opes = deepcopy(kcore_conn_opes)
    kcore_redu_num_opes = deepcopy(kcore_num_opes)
    for graph_idx in range(num_kcore_graph):
        tmp_ope_ma_adj = kcore_graph_list[graph_idx][:num_opes, num_opes:num_opes+num_mas]  # [n_opes, n_mas]
        non_elig_node = torch.where(tmp_ope_ma_adj.sum(dim=1) == 0)[0].tolist()
        elig_node = torch.Tensor(remove_elements(list(range(num_opes)), non_elig_node)).long()
        elig_ope_ma_adj = tmp_ope_ma_adj.gather(0, elig_node[:, None].expand(-1, num_mas))  # [elig_n_opes, n_mas]
        
        kcore_redu_ope_ma_adj.append(elig_ope_ma_adj)
        # re-setup connected operations and num_opes
        for node_idx in non_elig_node:
            for job_idx, opes in kcore_conn_opes[graph_idx].items():
                if node_idx in opes:
                    kcore_redu_conn_opes[graph_idx][job_idx].remove(node_idx)
                    kcore_redu_num_opes[graph_idx][job_idx] -= 1
        assert elig_ope_ma_adj.size(0) == sum(kcore_redu_num_opes[graph_idx])
        kcore_redu_num_opes[graph_idx] = [elem for elem in kcore_redu_num_opes[graph_idx] if elem>0]
    return kcore_redu_ope_ma_adj, kcore_redu_conn_opes, kcore_redu_num_opes
    
    
def remove_elements(target_list, input_list):
    """
    Remove elements from the input list within the target list.

    :param target_list: The target list.
    :param input_list: The input list.
    :return: The updated target list.
    """
    return [elem for elem in target_list if elem not in input_list]

def get_kcore_graph(num_opes, num_mas, num_vehs, adj_mat, spmat=True, draw_fig=False):
    '''
    Input:
        adj_mat: [n_opes, n_opes + n_mas + n_vehs]
    : return:
        k_core_graph_list = [k_core_graph, ...] length: max_core_num 
        k_core_att_list = [ope_conn_att, ...]
            ope_conn_att: connected operation info corresponding kcore graph
        k_core_num_opes_list = [] list of the number of operations for the job
    '''
    num_nodes = num_opes + num_mas + num_vehs
    
    full_nodes_list = np.arange(num_nodes)
    graph = _get_nx_graph(num_opes, num_mas, num_vehs, adj_mat)
    
    
    core_num_dict = nx.core_number(graph)
    max_core_num = max(list(core_num_dict.values()))
    
    k_core_graph_list = []
    k_core_att_list = []
    k_core_num_opes_list = []
    for i in range(0, max_core_num+1):
        k_core_graph = nx.k_core(graph, k=i, core_number=core_num_dict)
        k_core_graph.add_nodes_from(full_nodes_list)
        
        if spmat:
            A = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=full_nodes_list)
        else:
            A = nx.to_numpy_array(k_core_graph, nodelist=full_nodes_list)
        
        ope_conn_att = connected_nodes(A[:num_opes, :num_opes]) # dict: {job_idx:[ope_idxes], ...}
        k_core_num_opes_list.append([len(opes) for job_idx, opes in ope_conn_att.items()])
        
        k_core_graph_list.append(A)
        k_core_att_list.append(ope_conn_att)
    
    return k_core_graph_list, k_core_att_list, k_core_num_opes_list


def connected_nodes(adj_matrix):
    """
    Find connected nodes in a graph represented by an adjacency matrix.

    :param adj_matrix: A NumPy array representing the adjacency matrix.
    :return: A dict of connected nodes.
        {
            0 (job_id): [ope_idxes],
            1: [ope_idxes],
            ...
        }
    """
    # Convert adjacency matrix to a NetworkX graph
    graph = nx.from_numpy_array(adj_matrix)

    # Find connected components and convert them to lists
    components = nx.connected_components(graph)
    connected_components = {key: list(val) for key, val in enumerate(components)}
    # connected_components = list(nx.connected_components(graph))

    return connected_components
    
def _generate_env( 
    env_paras, num_opes, num_mas, num_vehs, device, dynamic=None,
    proctime_per_ope_max=20, transtime_btw_ma_max=10, 
):
    case = CaseGenerator(
        num_opes, num_mas, num_vehs, device,
        proctime_per_ope_max, transtime_btw_ma_max,
        dynamic,
    )
    env = TFJSPEnv(case=case, env_paras=env_paras)
    return env 
    

def draw_graph(graph):
    """
    Draw the given graph using NetworkX and Matplotlib.

    :param graph: A NetworkX graph.
    """
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold', edge_color='gray')
    plt.show()
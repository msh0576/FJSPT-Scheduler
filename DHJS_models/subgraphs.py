import torch
import numpy as np
import pandas as pd
import networkx as nx

from DHJS_models.utils import get_kcore_graph

def get_core_adj_mat(adj_mat, num_core, num_opes, num_mas, num_vehs, device, spmat=False):
    '''
    : param adj_mat: [B, n_opes, n_nodes]
    : return batch_core_adj_mat: [B, num_core, n_opes, n_nodes]
    '''
    batch_size = adj_mat.size(0)
    num_nodes = num_opes + num_mas + num_vehs
    
    batch_core_adj_mat = []
    for batch in range(batch_size):
        k_core_graph_list = get_kcore_graph(num_opes, num_mas, num_vehs, adj_mat[batch], spmat=spmat)
        k_core_graph_list = k_core_graph_list[::-1] # reverse order: max core, (max-1) core, ... 1 core
        # core_size = len(k_core_graph_list)
        # if core_size < num_core:
        #     k_core_graph_list = _duplicate_last_value(k_core_graph_list, num_core-core_size)
        # else:
        #     k_core_graph_list = k_core_graph_list[:num_core]    # top (num_core) subgraphs
        if not spmat:
            k_core_graph_np = np.array(k_core_graph_list)
            k_core_graph_tensor = torch.from_numpy(k_core_graph_np).float().to(device)
        else:
            raise Exception('cannot handle spmat_format!')
        batch_core_adj_mat.append(k_core_graph_tensor)
    # batch_core_adj_mat = torch.stack(batch_core_adj_mat)

    return batch_core_adj_mat


def _duplicate_last_value(target_list, n):
    """
    Duplicate the last value of the list by a given number.

    :param target_list: The target list.
    :param n: The number of times to duplicate the last value.
    :return: The modified list with the last value duplicated.
    """
    if not target_list:
        return []

    last_value = target_list[-1]
    for _ in range(n):
        target_list.append(last_value)

    return target_list
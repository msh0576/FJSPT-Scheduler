
import torch
from torch_geometric.data import Data

def set_GraphData(
    num_opes, num_mas, num_vehs,
    nums_ope, ope_ma_adj,
    proc_time, trans_time,
    node_feat_dim, edge_feat_dim
):
    '''
    :param nums_ope: tensor [n_jobs,]: the number of operation for a job
    :param ope_ma_adj: tensor [n_opes, n_mas]:
    :param proc_time: [n_opes, n_mas]
    :param trans_time: [n_mas, n_mas]
    
    :return graph_data: 
        Data['edge_index_bias'] = {
            'ope_ope_edge_bias': 0
            'ope_ma_edge_bias': 
            'ma_veh_edge_bias':
            'ma_ma_edge_bias':
            'num_edges':
            'num_nodes':
        }
    '''
    n_node = num_opes + num_mas + num_vehs
    edge_index_bias = {}
    # === transform nums_ope into opes_list ===
    opes_list = make_opes_list(nums_ope)
    # === build ope-ope edge_index ===
    # [2, n_ope_edge], [n_ope_edge, edge_feat_dim]
    ope_edge_idx, ope_edge_weig = build_ope_edge_tensor(opes_list) 
    edge_index_bias['ope_ope_edge_bias'] = 0
    
    # === build ope-ma edge_index ===
    ma_edge_idx = build_edge_index_from_graph_matrix(
        proc_time, col_bias=num_opes, target='ma'
    )
    ma_edge_weig = get_EdgeAttr_from_EdgeIndex(
        ma_edge_idx, proc_time, row_bias=0, col_bias=num_opes
    )
    edge_index_bias['ope_ma_edge_bias'] = ope_edge_idx.size(1)
    
    # === build veh-ma edge_index ===
    init_veh_loc = torch.zeros(size=(num_vehs,), dtype=torch.long)
    offload_trans_time = trans_time.gather(1, init_veh_loc[None, :].expand(trans_time.size(0),-1)) # [n_mas, n_vehs]
    offload_trans_time = offload_trans_time+1
    veh_edge_idx = build_edge_index_from_graph_matrix(
        offload_trans_time, row_bias=num_opes, col_bias=num_opes+num_mas, target='veh'
    )
    veh_edge_weig = get_EdgeAttr_from_EdgeIndex(
        veh_edge_idx, offload_trans_time, row_bias=num_opes, col_bias=num_opes+num_mas
    )
    edge_index_bias['ma_veh_edge_bias'] = ope_edge_idx.size(1) + ma_edge_idx.size(1)
    
    
    # === build ma-ma edge_index ===
    ma_full_mat = torch.triu(trans_time, diagonal=1)
    ma_full_edge_idx = build_edge_index_from_graph_matrix(
        ma_full_mat, row_bias=num_opes, col_bias=num_opes, target='ma'
    )
    ma_full_edge_weig = get_EdgeAttr_from_EdgeIndex(
        ma_full_edge_idx, ma_full_mat, row_bias=num_opes, col_bias=num_opes
    )
    edge_index_bias['ma_ma_edge_bias'] = ope_edge_idx.size(1) + ma_edge_idx.size(1) + veh_edge_idx.size(1)
    edge_index_bias['num_edges'] = ope_edge_idx.size(1) + ma_edge_idx.size(1) + \
        veh_edge_idx.size(1) + ma_full_edge_idx.size(1)
    edge_index_bias['num_nodes'] = n_node
    # === make graph dataset ===
    edge_idx = torch.cat([ope_edge_idx, ma_edge_idx, veh_edge_idx, ma_full_edge_idx], dim=1)
    edge_attr = torch.cat(
        [ope_edge_weig, ma_edge_weig, veh_edge_weig, ma_full_edge_weig], dim=0
    ).long()   # [n_edge, 1]
    x = torch.arange(n_node, dtype=torch.long)[:, None].expand(-1, node_feat_dim)   # [n_node, Dim_node_feat]
    data = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr,
                edge_index_bias=edge_index_bias)
    return data

def comp_edge_index_bias(num_nodes_batch):
    '''
    :param num_nodes_batch: batch size: ex) [14, 14, 14]
    :return edge_index_bias_batch: [0, 14, 28]
    '''
    edge_index_bias = [0]
    for i in range(len(num_nodes_batch)-1):
        edge_index_bias.append(edge_index_bias[-1] + num_nodes_batch[i])
    return edge_index_bias

def get_EdgeAttr_from_EdgeIndexBatch(edge_index_batch, graph_matrix, row_bias=0, col_bias=0):
    '''
    :param edge_index_batch: tensor [3, n_edge_batch]: selected edge_index
    :param graph_matrix: [B, row, col]
    '''
    n_edge = edge_index_batch.size(1)
    edge_attr = torch.zeros(size=(n_edge,), dtype=torch.long)
    for e in range(0, n_edge, 2):
        edge = edge_index_batch[:, e]
        assert edge.size(0) == 3
        batch = edge[0]
        row = edge[1] - row_bias
        col = edge[2] - col_bias
        z = graph_matrix[batch, row, col]
        edge_attr[e] = z
        edge_attr[e+1] = z  # for bidirection edge
    return edge_attr


def get_EdgeAttr_from_EdgeIndex(edge_index, graph_matrix, row_bias=0, col_bias=0):
    '''
    :param edge_index: [2, n_edge]
    :param graph_matrix: ex) [n_opes, n_mas]
    
    :return edge_attr: [n_edge]
    '''
    n_edge = edge_index.size(1)
    edge_attr = torch.zeros(size=(n_edge,), dtype=torch.float)
    for e in range(0, n_edge, 2):  # 짝수번째 edge due to bidirection edge_index
        edge = edge_index[:, e]
        assert edge.size(0) == 2
        row = edge[0] - row_bias
        col = edge[1] - col_bias
        
        z = graph_matrix[row, col]
        edge_attr[e] = z
        edge_attr[e+1] = z  # for bidirection edge
    return edge_attr

def build_edge_index_from_graph_matrix(graph_matrix, row_bias=0, col_bias=0, target='ma'):
    '''
    positive value element in the matrix is regarded by connected edge
    
    :param graph_matrix: [n_opes, n_mas]
    :param col_bias: ma_idx is biased by n_opes
    
    :return edge_index: [2, n_edges]
    '''
    row, col = graph_matrix.shape
    edge_list = []
    edge_weig_list = []

    for i in range(row):
        for j in range(col):
            if target == 'ma':
                if graph_matrix[i, j] > 0:
                    edge_list.append((i+row_bias, j+col_bias))
                    edge_list.append((j+col_bias, i+row_bias))
                    # edge_weig_list.append(graph_matrix[i,j])
                    # edge_weig_list.append(graph_matrix[i,j])
            elif target == 'veh':
                if graph_matrix[i, j] >= 0:
                    edge_list.append((i+row_bias, j+col_bias))
                    edge_list.append((j+col_bias, i+row_bias))
                    # edge_weig_list.append(graph_matrix[i,j])
                    # edge_weig_list.append(graph_matrix[i,j])
            else:
                raise Exception('target error in build_edge_index_from_graph_matrix()')

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # edge_weig = torch.tensor(edge_weig_list, dtype=torch.long)
    return edge_index


def build_ope_edge_tensor(input_list, edge_feat_dim=1):
    '''
    :param input_list [n_jobs, opes]
        ex) [[0, 1, ,2], [3, 4], [5]]
    :return edge_tensor: [2, n_edge]
    :return edge_weight: [n_edge,]
    '''
    edge_list = []

    for job in input_list:
        for i in range(len(job) - 1):
            edge_list.append((job[i], job[i + 1]))
            edge_list.append((job[i + 1], job[i]))

    edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    n_edges = edge_tensor.size(1)
    edge_weight = torch.ones(size=(n_edges,))
    return edge_tensor, edge_weight

def make_opes_list(nums_ope):
    '''
    :param nums_ope: tensor [n_jobs]
    :return opes_list: list of job-opeartion indexes
    '''
    nums_ope_list = nums_ope.tolist()
    opes_list = []
    prev_num_ope = 0
    for job_idx, num_ope in enumerate(nums_ope_list):
        if job_idx == 0:
            opes_list.append(list(range(num_ope)))
        else:
            prev_num_ope += nums_ope_list[job_idx-1]
            opes_list.append(list(range(prev_num_ope, prev_num_ope+num_ope)))
    return opes_list
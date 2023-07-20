import torch
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import numpy as np

class PositionEncoding(object):
    def apply_to(self, dataset):
        dataset.abs_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.abs_pe_list.append(pe)

        return dataset

    def batch_apply_to(self, dataset_batch):
        pe = self.compute_pe(dataset_batch)
        dataset_batch.abs_pe = pe

        return dataset_batch


class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def tmp_compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()
    
    def compute_pe(self, edge_index, edge_attr, num_nodes):
        edge_index, edge_attr = utils.get_laplacian(
            edge_index, edge_attr, normalization=self.normalization,
            num_nodes=num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()
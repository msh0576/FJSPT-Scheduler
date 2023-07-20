import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GINEConv
from torch.nn import ModuleList, Sequential, Linear, ReLU

from DHJS_models.encoding_block import EncodingBlock_Base

class TransformerEncoder(nn.Module):
    def __init__(self, 
        embed_dim, num_layers=3, **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(embed_dim, embed_dim),
                ReLU(),
                Linear(embed_dim, embed_dim),
            )
            conv = GPSConv(embed_dim, GINEConv(nn), heads=4, attn_dropout=0.5)
            self.layers.append(conv)
    
    def forward(self, x, edge_index, edge_attr, batch=None, pe=None):
        # x = self.node_emb(x.squeeze(-1)) + self.pe_lin(pe)
        if pe is not None:
            x = x + pe

        for layer in self.layers:
            x = layer(x, edge_index, batch, edge_attr=edge_attr)
        # x = global_add_pool(x, batch)
        # return self.lin(x)
        return x
    

        
        
        


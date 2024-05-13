import torch 
import torch.nn as nn
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import to_scipy_sparse_matrix

'''
neighbors_sum = torch.sparse.mm(adj_list , emb_matrix[0])
neighbors_sum = neighbors_sum.view(batch_size , neighbors_sum.shape[0] , neighbors_sum.shape[1])
'''

class embedding_network(nn.Module):
    
    def __init__(self , in_dim = 64, out_dim = 64 , T = 4, device = None , init_factor = 10 , w_scale = 0.01 , init_method = 'normal'):
        super().__init__()
        self.emb_dim = out_dim
        self.T = T
        self.W1 = nn.Linear(in_dim, out_dim , bias = False)
        self.W2 = nn.Linear(out_dim , out_dim , bias = False)
        self.W3 = nn.Linear(out_dim , out_dim , bias = False)
        self.W4 = nn.Linear( 1 , out_dim , bias = False)
        
        std = 1/np.sqrt(out_dim)/init_factor
        
        for W in [self.W1 , self.W2 , self.W3 , self.W4]:
            if init_method == 'normal':
                nn.init.normal_(W.weight , 0.0 , w_scale)
            else:
                nn.init.uniform_(W.weight , -std , std)
        self.device = device
        self.relu = nn.ReLU()
        
    def forward(self, Xv, edge_index):
        device = self.device
        n_vertex = Xv.shape[0]

        graph = to_scipy_sparse_matrix(edge_index)
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = graph.shape
        graph = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
        graph_edge = torch.unsqueeze(graph , 2) # n x n => n x n x 1

        emb_matrix = torch.zeros([n_vertex, self.emb_dim]).type(torch.FloatTensor)
        if 'cuda' in Xv.type():
            if device == None:
                emb_matrix = emb_matrix.cuda()
            else:
                emb_matrix = emb_matrix.cuda(device)

        for t in range(self.T):
            neighbor_sum = torch.sparse.mm(graph, emb_matrix)
            v1 = self.W1(Xv)                            # n x p => n x p
            v2 = self.W2(neighbor_sum)                  # n x p => n x p
            v3 = self.W4(graph_edge.to_dense())                    # n x n x 1 => n x n x p
            v3 = self.relu(v3)
            v3 = self.W3(torch.sum(v3 , 1))             # n x n x p => n x p
            
            #v = v1 + v2 + v3
            v1 = torch.add(v1 , v2)
            v = torch.add(v1 , v3)
            emb_matrix = self.relu(v)
        
        return emb_matrix
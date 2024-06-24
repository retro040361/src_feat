import torch
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes

def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    print(deg)
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights

edge = [[0,0,0,0,2,3],[1,2,3,4,3,4]]
edge_index = torch.tensor(edge)
out = degree_drop_weights(edge_index)
print(out)

import torch
from torch_geometric.utils import degree

def edge_weight_to_adj(edge_index, edge_weight, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    print(num_nodes)
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_weight.dtype, device=edge_weight.device)
    for i in range(len(edge_index[0])):
        # print(i)
        adj[edge_index[0][i]][edge_index[1][i]] = edge_weight[i]
        adj[edge_index[1][i]][edge_index[0][i]] = edge_weight[i]
    return adj

# # Example usage:
# edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Example edge index
# edge_weight = torch.tensor([0.5, 0.8, 1.0, 0.3])  # Example edge weight
# adj_matrix = edge_weight_to_adj(edge_index, edge_weight)

print("Adjacency Matrix:")
print(edge_weight_to_adj(edge_index, out))
print(torch.sigmoid(edge_weight_to_adj(edge_index, out)))

Z_o1 = encoder1(features, edge_index) 
Z_o2 = encoder2(features, edge_index) 
hidden_repr_o1 = encoder1.Z
hidden_repr_o2 = encoder2.Z

Z_t = encoder1(features, aug_edge_index)    # Z topology
Z_f = encoder2(aug_features, edge_index)

# AttnFusion
# attn_ft = AttentionModule(hidden2, num_nodes)
alpha_f, alpha_t = attn_ft(Z_f,Z_t)
Zft = alpha_f*Z_f + alpha_t*Z_t
alpha_o1, alpha_o2 = attn_ft(Z_o1,Z_o2)
Zo = alpha_o1*Z_o1 + alpha_o2*Z_o
# Loss
loss = torch.mean((Z_o1 - Z_o2) ** 2) # align loss
loss += inter_view_CL_loss(device, Zo, Zft, adj_label, delta, temperature)
if(loss_ver=="nei"):
    intra_CLft = inter_view_CL_loss(device, Zft, Zft, adj_label, gamma, temperature)
    intra_CLo = inter_view_CL_loss(device, Zo, Zo, adj_label, gamma, temperature)
else:
    intra_CLft = intra_view_CL_loss(device, Zft, adj_label, gamma, temperature)
    intra_CLo = intra_view_CL_loss(device, Zo, adj_label, gamma, temperature)
    
loss += intra_CLo
Aft_pred = dot_product_decode(Zft)
Ao_pred = dot_product_decode(Zo)
recon_loss_ft = loss_function(Aft_pred, adj_label, encoder1.mean, encoder1.logstd, norm, weight_tensor, alpha, beta, train_mask)
recon_loss_o = loss_function(Ao_pred, adj_label, encoder1.mean, encoder1.logstd, norm, weight_tensor, alpha, beta, train_mask)
        
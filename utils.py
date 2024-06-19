from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from model import dot_product_decode

from sklearn.neighbors import KernelDensity, kneighbors_graph
from scipy.stats import gaussian_kde, vonmises
from sklearn.preprocessing import normalize
from matplotlib import colors
import scipy.stats as st

import copy
import random
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected, add_remaining_self_loops, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.preprocessing import QuantileTransformer
# from sklearn.neighbors import kneighbors_graph
from torch.nn.functional import cosine_similarity
from torch_scatter import scatter_add
def vMF_KDE(dataset_str, Z):
    def nomalize_embedding(Z, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(Z, order, axis))
        l2[l2==0] = 1
        unit = Z / np.expand_dims(l2, axis)
        return unit

    transform = TSNE  # PCA
    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    emb_transformed = nomalize_embedding(emb_transformed)
    x = emb_transformed[0]
    y = emb_transformed[1]
    arc = np.arctan2(y, x)

    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.hist(arc, bins = 60)

    # kappa, loc, scale = vonmises.fit(arc, fscale=1)
    # mu, var, skew, kurt = vonmises.stats(kappa, moments='mvsk')
    # x = np.linspace(-np.pi, np.pi, num=51)
    # y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))  
    # plt.plot(x, y, linewidth=2, color='r')

    plt.show()
    plt.savefig(f'pic/{dataset_str}_ARC_KDE.png')
    plt.clf()

def gaussion_KDE(dataset_str, Z):

    def nomalize_embedding(Z, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(Z, order, axis))
        l2[l2==0] = 1
        unit = Z / np.expand_dims(l2, axis)
        return unit

    transform = TSNE  # PCA
    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    norm_emb_transformed = nomalize_embedding(emb_transformed)

    x = norm_emb_transformed[0]
    y = norm_emb_transformed[1]

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    z_norm = (z-min(z))/(max(z)-min(z))

    fig, ax = plt.subplots(figsize=(7,7))

    n = x.shape[0]
    c = 0.2
    colors = (1. - c) * plt.get_cmap("GnBu")(np.linspace(0., 1., n)) + c * np.ones((n, 4))
    ax.scatter(x, y, c=colors, s=30, alpha=z, marker='o')

    # ax.scatter(x, y, c=z, s=30, alpha=0.7, marker='o',cmap="GnBu")

    # ax.hexbin(x, y, cmap='Greens', gridsize=80)

    ############## Contour Version ################
    # xmin, xmax = -1.4, 1.4
    # ymin, ymax = -1.4, 1.4
    # # Peform the kernel density estimate
    # xx, yy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    # positions = np.vstack([xx.ravel(), yy.ravel()])
    # values = np.vstack([x, y])
    # kernel = st.gaussian_kde(values)
    # f = np.reshape(kernel(positions).T, xx.shape)
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.axis('equal')
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # # Contourf plot
    # cfset = ax.contourf(xx, yy, f, cmap='Greens') #, levels = [0.2, 0.3], extend='max')
    # ## Or kernel density estimate plot instead of the contourf plot
    # # ax.imshow(f, cmap='Greens', extent=[xmin, xmax, ymin, ymax])
    # # Contour plot
    # cset = ax.contour(xx, yy, f, colors='k')
    # # Label plot
    # ax.clabel(cset, inline=1, fontsize=10)
    # ax.set_xlabel('features')
    # ax.set_ylabel('features')
    ######################################################

    ##################################################
    # kernel = "gaussian"
    # X = norm_emb_transformed[0][:, np.newaxis]
    # X_plot = np.linspace(-1, 1, 10)[:, np.newaxis]
    # X_kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X) # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    # X_log_dens = X_kde.score_samples(X_plot)

    # fig, ax = plt.subplots()
    # ax.plot(
    #     X_plot[:, 0],
    #     np.exp(X_log_dens),
    #     color="navy",
    #     lw=2,
    #     linestyle="-", # label="kernel = '{0}'".format(kernel),
    # )
    # # ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")
    # # ax.set_xlim(-4, 9)
    # # ax.set_ylim(-0.02, 0.4)
    ##################################################

    plt.show()
    plt.savefig(f'pic/{dataset_str}_KDE.png')
    plt.clf()

def draw_adjacency_matrix_edge_weight_distribution(dataset_str, epoch, graph_type, adj):
    edge_weight_distribution = [
    torch.sum(adj == 0).item(),
    torch.sum((adj > 0.0) & (adj < 0.1)).item(),
    torch.sum((adj >= 0.1) & (adj < 0.2)).item(),
    torch.sum((adj >= 0.2) & (adj < 0.3)).item(),
    torch.sum((adj >= 0.3) & (adj < 0.4)).item(),
    torch.sum((adj >= 0.4) & (adj < 0.5)).item(),
    torch.sum((adj >= 0.5) & (adj < 0.6)).item(),
    torch.sum((adj >= 0.6) & (adj < 0.7)).item(),
    torch.sum((adj >= 0.7) & (adj < 0.8)).item(),
    torch.sum((adj >= 0.8) & (adj < 0.9)).item(),
    torch.sum((adj >= 0.9) & (adj < 1.0)).item(),
    torch.sum(adj == 1.0).item()]
    plt.plot(range(12), edge_weight_distribution, label = "edge_weight", marker="o")
    plt.xticks(ticks = range(12), labels = ['0.0', '<0.1', '<0.2', '<0.3', '<0.4', '<0.5', '<0.6', '<0.7', '<0.8', '<0.9', '<1.0', '1.0'])
    for i, j in zip(range(12), edge_weight_distribution):
        plt.annotate(str(j),xy=(i,j), fontsize = 7)
    plt.title(f'{dataset_str} {graph_type} epoch {epoch}')
    plt.savefig(f'pic/edge_weight_distribution/{dataset_str}/{graph_type}_epoch{epoch}.png')
    plt.clf()

def Plot(dataset_str, roc_history, modification_ratio_history):
    # plt.ylim(0.98, 0.99)
    plt.plot(roc_history, label = "roc_history")
    plt.legend()
    plt.savefig(f'pic/{dataset_str}_roc.png')
    plt.clf()
    plt.plot(modification_ratio_history, label = "modification_ratio_history")
    plt.legend()
    plt.savefig(f'pic/{dataset_str}_modification.png')
    plt.clf()

def Visualize(dataset_str, Z, labels):
    transform = TSNE  # PCA

    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    labels = torch.argmax(torch.tensor(labels), dim=1)
    emb_transformed["label"] = labels

    alpha = 0.7

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        s = 10,
        c = emb_transformed["label"].astype("category"),
        cmap="jet",
        alpha = alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title(f'{transform.__name__} visualization of embeddings for {dataset_str} dataset')
    plt.show()
    plt.savefig(f'pic/{dataset_str}_visualize.png')
    plt.clf()

def Visualize_with_edge(dataset_str, Z, labels, edge_index):
    
    transform = TSNE  # PCA
    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    labels = torch.argmax(torch.tensor(labels), dim=1)
    emb_transformed["label"] = labels

    node_num = Z.shape[0]
    edge_num = edge_index.shape[1]

    edge_x = []
    edge_y = []
    for i in range(edge_num):
        u = edge_index[0][i].item()
        v = edge_index[1][i].item()

        x0 = emb_transformed[0][u]
        y0 = emb_transformed[1][u]

        x1 = emb_transformed[0][v]
        y1 = emb_transformed[1][v]

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

    node_x = []
    node_y = []
    for i in range(node_num):
        x = emb_transformed[0][i]
        y = emb_transformed[1][i]

        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[],
            size=10,
            line_width=2))

    node_trace.marker.color = emb_transformed["label"].astype("category")

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

# def get_node_dist(graph):
#     """
#     Compute adjacent node distribution.
#     a.k.a calculate adjacency matrix
#     """
#     row, col = graph.edges()[0], graph.edges()[1]
#     num_node = graph.num_nodes()

#     dist_list = []
#     for i in range(num_node):
#         dist = torch.zeros([num_node], dtype=torch.float32, device=graph.device)
#         idx = row[(col==i)]
#         dist[idx] = 1
#         dist_list.append(dist)
#     dist_list = torch.stack(dist_list, dim=0)
#     return dist_list
def get_sim(embeds1, embeds2):
    # normalize embeddings across feature dimension
    embeds1 = F.normalize(embeds1)
    embeds2 = F.normalize(embeds2)
    sim = torch.mm(embeds1, embeds2.t())
    return sim

def neighbor_sampling(src_idx, dst_idx, node_dist, sim, 
                    max_degree, aug_degree):
    phi = sim[src_idx, dst_idx].unsqueeze(dim=1)
    phi = torch.clamp(phi, 0, 0.5)

    # print('phi', phi)
    mix_dist = node_dist[dst_idx]*phi + node_dist[src_idx]*(1-phi)

    new_tgt = torch.multinomial(mix_dist + 1e-12, int(max_degree)).to(phi.device)
    tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(phi.device)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = src_idx.repeat_interleave(aug_degree)
    return new_row, new_col

def degree_mask_edge(idx, sim, max_degree, node_degree, mask_prob):
    aug_degree = (node_degree * (1- mask_prob)).long().to(sim.device)
    sim_dist = sim[idx]

    # _, new_tgt = th.topk(sim_dist + 1e-12, int(max_degree))
    new_tgt = torch.multinomial(sim_dist + 1e-12, int(max_degree))
    tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(sim.device)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = idx.repeat_interleave(aug_degree)
    return new_row, new_col

def degree_aug(bias_Z, adj_labl, degree,num_nodes, edge_mask_rate_1, threshold, epoch):
    if(epoch < 200):
        return adj_labl, 1
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    
    max_degree = np.max(degree)
    src_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    rest_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    rest_node_degree = degree[degree>=threshold]    # head node
    
    sim = get_sim(bias_Z, bias_Z)
    sim = torch.clamp(sim, 0, 1)
    sim = sim - torch.diag_embed(torch.diag(sim))
    src_sim = sim[src_idx]

    dst_idx = torch.multinomial(src_sim + 1e-12, 1).flatten().to(device)    # tail node

    rest_node_degree = torch.LongTensor(rest_node_degree)
    degree_dist = scatter_add(torch.ones(rest_node_degree.size()), rest_node_degree).to(device)
    prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)

    aug_degree = torch.multinomial(prob, 1).flatten().to(device)

    new_row_mix_1, new_col_mix_1 = neighbor_sampling(src_idx, dst_idx, adj_labl, sim, max_degree, aug_degree)   # tail node add
    new_row_rest_1, new_col_rest_1 = degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_1)    # head node purify
    nsrc1 = torch.cat((new_row_mix_1, new_row_rest_1)).cpu()
    ndst1 = torch.cat((new_col_mix_1, new_col_rest_1)).cpu()
    
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[nsrc1, ndst1] = 1
    adj += torch.eye(num_nodes, dtype=torch.float32)
    adj = adj.to(device)
    return adj, 1

### 0619 modify 
def sparse_mx_to_edge_index(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    row = torch.from_numpy(sparse_mx.row.astype(np.int64))
    col = torch.from_numpy(sparse_mx.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
 
    return edge_index

def knn_graph(X, k=20, metric='minkowski'):
    X = X.cpu().detach().numpy()
    A = kneighbors_graph(X, n_neighbors=k, metric=metric)
    edge_index = sparse_mx_to_edge_index(A)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index


def merge_neighbors_efficient(adj_label, adj_knn, tail_idx, r, k):
    num_nodes = adj_label.size(0)
    tail_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    tail_mask[tail_idx] = True

    # 根据掩码提取尾节点的邻接信息
    tail_adj_label = adj_label[tail_mask]
    tail_adj_knn = adj_knn[tail_mask]

    # 计算每个尾节点应该保留的邻居数（r%和1-r%）
    num_neighbors_label = (tail_adj_label.sum(dim=1) * r).ceil().int()
    num_neighbors_knn = int((1 - r) * k) #(tail_adj_knn.sum(dim=1) * (1 - r)).int()

    # 创建新的邻接矩阵
    new_adj = adj_label.clone()

    # 随机选择原始邻居进行保留
    total_label_neighbors = tail_adj_label.sum(1)
            # probs_label[total_label_neighbors == 0] = 0  # 避免除以零 有可能會有這個問題 但遇到再說zz
    probs_label = tail_adj_label / total_label_neighbors[:, None]  # 归一化概率
    sampled_label = torch.multinomial(probs_label, num_neighbors_label.max(), replacement=True)
    keep_label_mask = torch.zeros_like(tail_adj_label).scatter_(1, sampled_label, 1)

    # 随机选择KNN邻居进行添加
    total_knn_neighbors = tail_adj_knn.sum(1)
    probs_knn = tail_adj_knn / total_knn_neighbors[:, None]  # 归一化概率
    sampled_knn = torch.multinomial(probs_knn, num_neighbors_knn, replacement=False)
    keep_knn_mask = torch.zeros_like(tail_adj_knn).scatter_(1, sampled_knn, 1)

    new_adj[tail_mask] = keep_label_mask + keep_knn_mask
    # new_adj[tail_mask] = keep_label_mask #[tail_mask]  # 更新原始邻居
    # new_adj[tail_mask] = keep_knn_mask #[tail_mask]   # 更新KNN邻居

    return tail_mask.cpu(), keep_label_mask.cpu(), keep_knn_mask.cpu() #new_adj

def purify_head_nodes_efficient(bias_Z, adj_matrix, head_idx, r, new_adj=None):
    # cos_sim_matrix = cosine_similarity(bias_Z, bias_Z)
    # normalized_cos_sim = cos_sim_matrix / cos_sim_matrix.sum(1, keepdim=True)
    # head_adj = adj_matrix[head_idx]
    # normalized_cos_sim_head = normalized_cos_sim[head_idx]
    # # # 对于每个头节点，找到所有邻居的索引
    # # neighbors_mask = head_adj > 0

    # # 使用余弦相似度和邻居掩码计算实际的相似度
    # # head_cos_sim = cos_sim_matrix[head_idx][:, neighbors_mask]
    # # head_cos_sim = normalized_cos_sim[head_idx][:, neighbors_mask]

    # # 计算需要保留的邻居数量
    # # num_neighbors = neighbors_mask.sum(dim=1)
    # # num_to_keep = (num_neighbors.float() * (1 - r)).long()
    # num_to_keep = (head_adj.sum(dim=1)*(1-r)).ceil().int()
    # # 对每行使用topk来找出相似度最高的邻居
    # sampled_indices = torch.multinomial(normalized_cos_sim_head, num_to_keep.max(), replacement=True)
    # # _, top_indices = torch.topk(head_cos_sim, k=num_to_keep.min().item(), dim=1, largest=True)

    # # 重置所有头节点的邻接信息
    # new_adj[head_idx] = 0
    # new_adj[head_idx] = new_adj[head_idx].scatter_(1, sampled_indices, 1)
    
    num_nodes = adj_matrix.size(0)
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_matrix.device)
    head_mask[head_idx] = True
    
    head_adj = adj_matrix[head_mask]
    num_to_keep = (head_adj.sum(dim=1)*(1-r)).ceil().int()
    
    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2*abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1)[:,None]
    normalized_sim_head = normalized_sim[head_mask]
    print(normalized_sim_head)
    sampled_indices = torch.multinomial(normalized_sim_head, num_to_keep.max(), replacement=True)
    keep_label_mask = torch.zeros_like(head_adj).scatter_(1, sampled_indices, 1)
    
    # new_adj[head_mask] = 0
    # new_adj[head_mask] = keep_label_mask
    
    

    return head_mask.cpu(), keep_label_mask.cpu()

def degree_aug_v2(bias_Z, adj_label, degree,num_nodes, edge_flip_rate, threshold, epoch):
    if(epoch < 200):
        return adj_label, 1
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    kg_edge_index = knn_graph(bias_Z, k=threshold, metric='euclidean')
    adj_knn = torch.sparse_coo_tensor(kg_edge_index, torch.ones(kg_edge_index.shape[1]), (num_nodes, num_nodes)).to_dense().to(device)

    head_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    
    # promote low degree node
    tail_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    tail_node_degree = degree[degree<threshold]
    idx_degree_pair = np.column_stack((tail_idx.cpu().numpy(), tail_node_degree))
    # transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    # normalized_tail_degrees = transformer.fit_transform(idx_degree_pair[:, 1].reshape(-1, 1)).flatten() + threshold
    # normalized_idx_degree_pair = np.column_stack((idx_degree_pair[:, 0], normalized_tail_degrees))    # pair[.][0] = idx, pair[.][1] = normalize degree 
    
    tail_mask, tail_keep_label_mask, keep_knn_mask = merge_neighbors_efficient(adj_label, adj_knn,tail_idx, edge_flip_rate, threshold)
    head_mask, head_keep_label_mask = purify_head_nodes_efficient(bias_Z, adj_label, head_idx, edge_flip_rate)
    
    new_adj=adj_label.clone()
    new_adj[tail_mask.to(device)] = tail_keep_label_mask.to(device) + keep_knn_mask.to(device)
    new_adj[head_mask.to(device)] = head_keep_label_mask.to(device)
    
    mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    new_adj.masked_fill_(mask, 0)
    # print(new_adj)
    # demote high degree node
    # num_nodes = adj_label.size(0)
    # head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    # head_mask[head_idx] = True
    # head_adj_matrix = adj_[head_idx]
    
    return new_adj, 1



def Graph_Modify_Constraint(bias_Z, original_graph, k, bound):
    print("original function")
    aug_graph = dot_product_decode(bias_Z)
    constrainted_new_graph = original_graph.clone()
    # print(type(constrainted_new_graph))
    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    difference = torch.abs(aug_graph - original_graph) # difference[difference < 0] = 1.0 ###
    difference += torch.eye(difference.shape[0]).to(difference.device) * 2

    values, indices = torch.topk(difference.flatten(), k, largest = False)
    indices_mask = ((values >= 0.0) & (values <= bound))
    mask = indices[indices_mask].type(torch.long)
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]

    print(f'Avg modified difference value: {torch.mean(difference.flatten()[mask])}')
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    # print(type(constrainted_new_graph))
    return constrainted_new_graph, mask.shape[0] / constrainted_new_graph.flatten().shape[0]

def Graph_Modify_Constraint_feat(bias_Z, original_graph, k, bound, feat_sim):
    print("local structure modified function")
    aug_graph = dot_product_decode(bias_Z)
    constrainted_new_graph = original_graph.clone().to('cpu')

    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    difference = torch.abs(aug_graph - original_graph) # difference[difference < 0] = 1.0 ###
    difference += torch.eye(difference.shape[0]).to(difference.device) * 2
    device = difference.device
    difference = difference.to('cpu')
    # for i in range(difference.shape[0]):
    #     for j in range(difference.shape[1]):
    #         if difference[i, j] < 0.5:
    #             if common_neighbors_count[i, j] > cn_threshold:
    #                 constrainted_new_graph[i, j] = 1
    #             else:
    #                 constrainted_new_graph[i, j] = 0
    print(difference.device, constrainted_new_graph.device)
    link_mask = (difference < bound) & (feat_sim > 0.6) & (constrainted_new_graph == 0)
    unlink_mask = (difference < bound) & (feat_sim <= 0.6) & (constrainted_new_graph == 1) 
    link_mask = link_mask.type(torch.bool)
    unlink_mask = unlink_mask.type(torch.bool)

    link_indices = link_mask.nonzero(as_tuple=False)
    unlink_indices = unlink_mask.nonzero(as_tuple=False)
    total_possible_changes = link_indices.size(0) + unlink_indices.size(0)
    if total_possible_changes < k:
        print(f"total possible changes less than k: {total_possible_changes}")
        # print()
    if(link_indices.size(0)==0):
        print("indice shape = 0")
        return constrainted_new_graph.to(device), 0
    k_link = min(link_indices.size(0), k // 2)
    k_unlink = min(unlink_indices.size(0), k // 2)
    print(f"link indice size:{link_indices.size(0)}, unlink indice size:{unlink_indices.size(0)}")
    
    chosen_link_indices = link_indices[torch.randperm(link_indices.size(0))[:k_link]]
    print(f"chosen link indice #:{k_link}")
    print(chosen_link_indices[:5])
    chosen_unlink_indices = unlink_indices[torch.randperm(unlink_indices.size(0))[:k_unlink]]
    # chosen_indices = torch.cat([chosen_link_indices, chosen_unlink_indices])
    # constrainted_new_graph[chosen_indices] = 1 - constrainted_new_graph[chosen_indices]
    constrainted_new_graph[chosen_link_indices] = 1
    constrainted_new_graph[chosen_unlink_indices] = 0

    # constrainted_new_graph[link_mask] = 1
    # constrainted_new_graph[unlink_mask] = 0

    return constrainted_new_graph.to(device), chosen_link_indices.shape[0] / constrainted_new_graph.flatten().shape[0]

def Graph_Modify_Constraint_local(bias_Z, original_graph, k, bound, common_neighbors_count, cn_threshold):
    print("local structure modified function")
    aug_graph = dot_product_decode(bias_Z)
    constrainted_new_graph = original_graph.clone().to('cpu')

    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    difference = torch.abs(aug_graph - original_graph) # difference[difference < 0] = 1.0 ###
    difference += torch.eye(difference.shape[0]).to(difference.device) * 2
    device = difference.device
    difference = difference.to('cpu')
    # for i in range(difference.shape[0]):
    #     for j in range(difference.shape[1]):
    #         if difference[i, j] < 0.5:
    #             if common_neighbors_count[i, j] > cn_threshold:
    #                 constrainted_new_graph[i, j] = 1
    #             else:
    #                 constrainted_new_graph[i, j] = 0
    print(difference.device, constrainted_new_graph.device)
    link_mask = (difference < bound) & (common_neighbors_count > cn_threshold) & (constrainted_new_graph == 0)
    unlink_mask = (difference < bound) & (common_neighbors_count <= cn_threshold) & (constrainted_new_graph == 1) 
    link_mask = link_mask.type(torch.bool)
    unlink_mask = unlink_mask.type(torch.bool)

    link_indices = link_mask.nonzero(as_tuple=False)
    unlink_indices = unlink_mask.nonzero(as_tuple=False)
    total_possible_changes = link_indices.size(0) + unlink_indices.size(0)
    if total_possible_changes < k:
        print(f"total possible changes less than k: {total_possible_changes}")
        # print()
    
    k_link = min(link_indices.size(0), k // 2)
    k_unlink = k - k_link
    chosen_link_indices = link_indices[torch.randperm(link_indices.size(0))[:k_link]]
    print(f"chosen link indice #:{k_link}")
    print(chosen_link_indices[:5])
    chosen_unlink_indices = unlink_indices[torch.randperm(unlink_indices.size(0))[:k_unlink]]
    # chosen_indices = torch.cat([chosen_link_indices, chosen_unlink_indices])
    # constrainted_new_graph[chosen_indices] = 1 - constrainted_new_graph[chosen_indices]
    constrainted_new_graph[chosen_link_indices] = 1
    constrainted_new_graph[chosen_unlink_indices] = 0

    # constrainted_new_graph[link_mask] = 1
    # constrainted_new_graph[unlink_mask] = 0

    return constrainted_new_graph.to(device), chosen_link_indices.shape[0] / constrainted_new_graph.flatten().shape[0]

def Graph_Modify_Constraint_exp(bias_Z, original_graph, k, bound):
    print("Thm experiment")
    aug_graph = dot_product_decode(bias_Z)
    constrainted_new_graph = original_graph.clone()
    # print(type(constrainted_new_graph))
    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    difference = torch.abs(aug_graph - original_graph) # difference[difference < 0] = 1.0 ###
    difference += torch.eye(difference.shape[0]).to(difference.device) * 2
    
    edge_index = original_graph.to_sparse().indices()
    degree_weight = degree_drop_weights(edge_index)
    degree_weight_adj = edge_weight_to_adj(edge_index, degree_weight)
    degree_weight_adj_sc = torch.sigmoid(degree_weight_adj)
    # degree weight big => node centrality small => remove probability big 
    # difference big => not-well learned yet => choose smallest k
    difference = (0.4*difference + 0.6*(1-degree_weight_adj_sc))   
    # _, indices = torch.topk(degree_weight_adj_sc.flatten(), k, largest=True)
    # values = difference.flatten()[indices]
    values, indices = torch.topk(difference.flatten(), k, largest = False)
    indices_mask = ((values >= 0.0) & (values <= bound))
    mask = indices[indices_mask].type(torch.long)
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]

    print(f'Avg modified difference value: {torch.mean(difference.flatten()[mask])}')
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    # print(type(constrainted_new_graph))
    return constrainted_new_graph, mask.shape[0] / constrainted_new_graph.flatten().shape[0]

def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights

def edge_weight_to_adj(edge_index, edge_weight, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    print(num_nodes)
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_weight.dtype, device=edge_weight.device)
    for i in range(len(edge_index[0])):
        # print(i)
        adj[edge_index[0][i]][edge_index[1][i]] = edge_weight[i]
        adj[edge_index[1][i]][edge_index[0][i]] = edge_weight[i]
    return adj

def aug_random_edge(input_adj, drop_percent=0.2):
    device = input_adj.device

    percent = drop_percent / 2
    indices = input_adj.nonzero()
    row_idx, col_idx = indices[0], indices[1]
    #row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i].item(), col_idx[i].item()))

    single_index_list = []
    to_remove = []  # List to store elements to remove
    for i in list(index_list):
        single_index_list.append(i)
        to_remove.append((i[1], i[0]))  # Store the pair to remove

    # Remove elements from index_list
    index_list = [pair for pair in index_list if pair not in to_remove]


    edge_num = len(row_idx) // 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = input_adj.clone().to_dense()

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0], single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1], single_index_list[i][0]] = 0

    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0], i[1]] = 1
        aug_adj[i[1], i[0]] = 1

    aug_adj = aug_adj.to(device)
    aug_adj = sp.csr_matrix(aug_adj.cpu().numpy())
    aug_adj = torch.tensor(aug_adj.todense(), device=device)

    return aug_adj, drop_percent

# def random_edge_modification(graph, k):
#     k_add = int(random.random*k)
#     k_delete = k - k_add
#     aug_graph = graph.clone()
#     num_edges = aug_graph.edge_index.size(1)
    
#     # Randomly select k_add edges to add
#     add_indices = torch.randint(num_edges, size=(k_add,))
#     add_edges = graph.edge_index[:, add_indices]
    
#     # Randomly select k_delete edges to delete
#     delete_indices = torch.randint(num_edges, size=(k_delete,))
    
#     # Create a mask to keep the non-deleted edges
#     mask = torch.ones(num_edges, dtype=torch.bool)
#     mask[delete_indices] = False
    
#     # Apply modifications to the graph
#     modified_edge_index = graph.edge_index[:, mask]
#     modified_graph = Data(x=graph.x, edge_index=modified_edge_index)
    
#     # Add the new edges to the modified graph
#     modified_graph.edge_index = torch.cat([modified_graph.edge_index, add_edges], dim=1)
    
#     return modified_graph




'''
        # ### Add top k/2 percent edges, remove top k/2 percent edges ###
        # difference = aug_graph - original_graph
        # add_values, add_indices = torch.topk(difference.flatten(), int(k/2), largest = True)
        # remove_values, remove_indices = torch.topk(difference.flatten(), int(k/2), largest = False)

        # add_mask = add_indices[add_values > 0]
        # remove_mask = remove_indices[remove_values < 0]
        # assert torch.all(constrainted_new_graph.flatten()[add_mask] == 0) # assert torch.all(add_values > 0)
        # assert torch.all(constrainted_new_graph.flatten()[remove_mask] == 1) # assert torch.all(remove_values < 0)

        # constrainted_new_graph.flatten()[add_mask] = 1 - constrainted_new_graph.flatten()[add_mask]
        # constrainted_new_graph.flatten()[remove_mask] = 1 - constrainted_new_graph.flatten()[remove_mask]

        # if output_device == 'cpu':
        #     ret.append(constrainted_new_graph.cpu())
        # else:
        #     ret.append(constrainted_new_graph)

        ###
        # # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2
        # difference = torch.abs(aug_graph - original_graph)
        # indices = torch.arange(constrainted_new_graph.flatten().shape[0])
        # indices_mask = ((difference.flatten() >= 0.0) & (difference.flatten() <= 0.2))
        # mask = indices[indices_mask].type(torch.long)
        # print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
        # constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
        # modification_ratio.append(mask.shape[0] / constrainted_new_graph.flatten().shape[0])
        # if output_device == 'cpu':
        #     ret.append(constrainted_new_graph.cpu())
        # else:
        #     ret.append(constrainted_new_graph)
        ###

        ###
        # # Modify/Flip the most accuate edges
        # difference = torch.abs(aug_graph - original_graph)
        # values, indices = torch.topk(difference.flatten(), k)
        # print(f'avg modified difference value: {torch.mean(values)}')
        # print(f'Modify percentage: {indices.shape[0] / constrainted_new_graph.flatten().shape[0]}')
        # constrainted_new_graph.flatten()[indices] = 1 - constrainted_new_graph.flatten()[indices]
        # modification_ratio.append(indices.shape[0] / constrainted_new_graph.flatten().shape[0])
        # if output_device == 'cpu':
        #     ret.append(constrainted_new_graph.cpu())
        # else:
        #     ret.append(constrainted_new_graph)
        ###

        ###
        # # adding the most accuate edges, error ranging from 0.0 ~ 0.2
        # difference = aug_graph - original_graph
        # indices = torch.arange(constrainted_new_graph.flatten().shape[0])
        # indices_mask = ((difference.flatten() >= 0.0) & (difference.flatten() <= 0.2))
        # mask = indices[indices_mask].type(torch.long)
        # print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
        # constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
        # if output_device == 'cpu':
        #     ret.append(constrainted_new_graph.cpu())
        # else:
        #     ret.append(constrainted_new_graph)
        ###

        ###
        # # removing the most accuate edges, error ranging from 0.0 ~ 0.2
        # difference = original_graph - aug_graph
        # indices = torch.arange(constrainted_new_graph.flatten().shape[0])
        # indices_mask = ((difference.flatten() >= 0.0) & (difference.flatten() <= 0.2))
        # mask = indices[indices_mask].type(torch.long)
        # print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
        # constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
        # if output_device == 'cpu':
        #     ret.append(constrainted_new_graph.cpu())
        # else:
        #     ret.append(constrainted_new_graph)
        ###

        ###
        # ### Add accurate k/2 percent edges, remove accurate k/2 percent edges ###
        # difference = aug_graph - original_graph
        # indices = torch.arange(constrainted_new_graph.flatten().shape[0])
        # add_indices_mask = (difference.flatten() >= 0.0)
        # mask = indices[add_indices_mask].type(torch.long)
        # add_values, add_indices = torch.topk(difference.flatten()[mask], int(k/2), largest = False)
        # constrainted_new_graph.flatten()[add_indices] = 1 - constrainted_new_graph.flatten()[add_indices]
        
        # difference = aug_graph - original_graph
        # indices = torch.arange(constrainted_new_graph.flatten().shape[0])
        # remove_indices_mask = (difference.flatten() <= 0.0)
        # mask = indices[remove_indices_mask].type(torch.long)
        # remove_values, remove_indices = torch.topk(difference.flatten(), int(k/2), largest = True)
        # constrainted_new_graph.flatten()[remove_indices] = 1 - constrainted_new_graph.flatten()[remove_indices]
        # print(f'Modify percentage: {(remove_indices.shape[0]+add_indices.shape[0]) / constrainted_new_graph.flatten().shape[0]}')
        # if output_device == 'cpu':
        #     ret.append(constrainted_new_graph.cpu())
        # else:
        #     ret.append(constrainted_new_graph)
        ###
'''
'''
    # g1: add edge, g2: remove edge, g3: add and remove edge
    constrainted_new_graph = original_graph.clone()
    difference = aug_graphs[0] - original_graph
    add_values, add_indices = torch.topk(difference.flatten(), k, largest = True)
    add_mask = add_indices[add_values > 0]
    assert torch.all(constrainted_new_graph.flatten()[add_mask] == 0)
    constrainted_new_graph.flatten()[add_mask] = 1 - constrainted_new_graph.flatten()[add_mask]
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    constrainted_new_graph = original_graph.clone()
    difference = aug_graphs[1] - original_graph
    remove_values, remove_indices = torch.topk(difference.flatten(), k, largest = False)
    remove_mask = remove_indices[remove_values < 0]
    assert torch.all(constrainted_new_graph.flatten()[remove_mask] == 1)
    constrainted_new_graph.flatten()[remove_mask] = 1 - constrainted_new_graph.flatten()[remove_mask]
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    constrainted_new_graph = original_graph.clone()
    difference = aug_graphs[2] - original_graph
    add_values, add_indices = torch.topk(difference.flatten(), int(k/2), largest = True)
    remove_values, remove_indices = torch.topk(difference.flatten(), int(k/2), largest = False)
    add_mask = add_indices[add_values > 0]
    remove_mask = remove_indices[remove_values < 0]
    assert torch.all(constrainted_new_graph.flatten()[add_mask] == 0)
    assert torch.all(constrainted_new_graph.flatten()[remove_mask] == 1)
    constrainted_new_graph.flatten()[add_mask] = 1 - constrainted_new_graph.flatten()[add_mask]
    constrainted_new_graph.flatten()[remove_mask] = 1 - constrainted_new_graph.flatten()[remove_mask]
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
'''
'''
    ###
    # g1: Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.1, g2: Modify/Flip the most accuate edges, error ranging from 0.1 ~ 0.2, g3: Modify/Flip the most accuate edges, error ranging from 0.2 ~ 0.3
    constrainted_new_graph = original_graph.clone()
    difference = torch.abs(aug_graphs[0] - original_graph)
    indices = torch.arange(constrainted_new_graph.flatten().shape[0])
    indeices_mask = ((difference.flatten() >= 0.0) & (difference.flatten() <= 0.1))
    mask = indices[indeices_mask].type(torch.long)
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    constrainted_new_graph = original_graph.clone()
    difference = torch.abs(aug_graphs[1] - original_graph)
    indices = torch.arange(constrainted_new_graph.flatten().shape[0])
    indeices_mask = ((difference.flatten() > 0.1) & (difference.flatten() <= 0.2))
    mask = indices[indeices_mask].type(torch.long)
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    constrainted_new_graph = original_graph.clone()
    difference = torch.abs(aug_graphs[2] - original_graph)
    indices = torch.arange(constrainted_new_graph.flatten().shape[0])
    indeices_mask = ((difference.flatten() > 0.2) & (difference.flatten() <= 0.3))
    mask = indices[indeices_mask].type(torch.long)
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    ###
'''
'''
    ###
    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    constrainted_new_graph = original_graph.clone()
    difference = torch.abs(aug_graphs[0] - original_graph)
    values, indices = torch.topk(difference.flatten(), int(k/3), largest = False)
    indices_mask = ((values >= 0.0) & (values <= 0.2))
    mask = indices[indices_mask].type(torch.long)
    print(f'Avg modified difference value: {torch.mean(difference.flatten()[mask])}')
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
    modification_ratio.append(mask.shape[0] / constrainted_new_graph.flatten().shape[0])
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    constrainted_new_graph = original_graph.clone()
    difference = torch.abs(aug_graphs[1] - original_graph)
    values, indices = torch.topk(difference.flatten(), int(k/3), largest = False)
    indices_mask = ((values >= 0.9) & (values <= 1.0))
    mask = indices[indices_mask].type(torch.long)
    print(f'Avg modified difference value: {torch.mean(difference.flatten()[mask])}')
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
    modification_ratio.append(mask.shape[0] / constrainted_new_graph.flatten().shape[0])
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    constrainted_new_graph = original_graph.clone()
    difference = torch.abs(aug_graphs[2] - original_graph)
    values, indices = torch.topk(difference.flatten(), int(k/3), largest = False)
    indices_mask = ((values >= 0.4) & (values <= 0.5))
    mask = indices[indices_mask].type(torch.long)
    print(f'Avg modified difference value: {torch.mean(difference.flatten()[mask])}')
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]
    modification_ratio.append(mask.shape[0] / constrainted_new_graph.flatten().shape[0])
    if output_device == 'cpu':
        ret.append(constrainted_new_graph.cpu())
    else:
        ret.append(constrainted_new_graph)
    ###
'''
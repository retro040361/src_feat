import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def process_head_node(idx, normalized_sim, num_to_keep, new_adj):
    current_sim = normalized_sim[idx]
    n_keep = num_to_keep.item()
    if n_keep > 0:
        # Use topk to find the indices of the highest similarity scores
        _, top_indices = torch.topk(current_sim, n_keep, largest=True, sorted=False)
        new_adj[idx, top_indices] = 1  # Set new connections in the adjacency matrix

def purify_head_nodes_v3p(bias_Z, adj_matrix, head_idx, r, new_adj):
    num_nodes = adj_matrix.size(0)
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_matrix.device)
    head_mask[head_idx] = True
    
    # sim_matrix = cosine_similarity(bias_Z, bias_Z)
    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2 * abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1, keepdim=True)
    head_adj = adj_matrix[head_idx]
    num_to_keep = (head_adj.sum(dim=1) * (1 - r)).ceil().int()
    
    new_adj[head_idx] = 0  # Reset adjacency connections for head nodes

    # Multi-threading with 4 threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_head_node, idx, normalized_sim, num_to_keep[i], new_adj) for i, idx in enumerate(head_idx)]
        for future in futures:
            future.result()  # Wait for all threads to complete

    return new_adj


def process_tail_node(i, idx, tail_adj_label, tail_adj_knn, r, new_adj, old_degrees, new_degrees):
    # 当前尾节点的原始邻居信息和KNN邻居信息
    current_label_neighbors = tail_adj_label[i]
    old_degrees[i]=current_label_neighbors.sum().item()
    current_knn_neighbors = tail_adj_knn[i]

    # 计算应保留和新增的邻居数
    num_to_keep_label = int((current_label_neighbors.sum() * (1-r)).ceil().item())
    num_to_keep_knn = int((current_knn_neighbors.sum() * r).ceil().item())

    # 随机选择原始邻居进行保留
    if current_label_neighbors.sum() > 0:
        all_label_indices = torch.where(current_label_neighbors > 0)[0]
        rand_indices = torch.randperm(all_label_indices.size(0))[:num_to_keep_label]
        keep_label_indices = all_label_indices[rand_indices]
        label_mask = torch.zeros_like(current_label_neighbors)
        label_mask[keep_label_indices] = 1
    else:
        label_mask = torch.zeros_like(current_label_neighbors)

    # 随机选择KNN邻居进行添加
    if current_knn_neighbors.sum() > 0:
        all_knn_indices = torch.where(current_knn_neighbors > 0)[0]
        rand_indices = torch.randperm(all_knn_indices.size(0))[:num_to_keep_knn]
        keep_knn_indices = all_knn_indices[rand_indices]
        knn_mask = torch.zeros_like(current_knn_neighbors)
        knn_mask[keep_knn_indices] = 1
    else:
        knn_mask = torch.zeros_like(current_knn_neighbors)

    # 更新邻接矩阵
    all_mask = label_mask + knn_mask
    new_adj[idx] = all_mask  # 使用逻辑或合并保留和新增的邻居
    new_degrees[i]=all_mask.sum().item()

def merge_neighbors_v3p(adj_label, adj_knn, tail_idx, r, k):
    num_nodes = adj_label.size(0)
    tail_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    tail_mask[tail_idx] = True

    # 提取尾节点的邻接信息
    tail_adj_label = adj_label[tail_mask]
    tail_adj_knn = adj_knn[tail_mask]

    # 创建新的邻接矩阵
    new_adj = adj_label.clone()
    old_degrees = [0]*tail_idx.size()[0]
    new_degrees = [0]*tail_idx.size()[0]

    # 多线程处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_tail_node, i, idx, tail_adj_label, tail_adj_knn, r, new_adj, old_degrees, new_degrees)
                   for i, idx in enumerate(tail_idx)]
        for future in futures:
            future.result()  # 等待所有线程完成

    print(f"[Tail Node] origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
    return new_adj


import os
import time
import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

from ogb.linkproppred import Evaluator
from tqdm import tqdm
from preprocessing import *
from model import VGNAE_ENCODER, VGAE_ENCODER, dot_product_decode, MLP, LogReg
from loss import loss_function, inter_view_CL_loss, intra_view_CL_loss, Cluster
# from utils import Visualize, Graph_Modify_Constraint, draw_adjacency_matrix_edge_weight_distribution, aug_random_edge, Graph_Modify_Constraint_exp
from utils import *
from input_data import CalN2V

from sklearn.metrics.pairwise import cosine_similarity

# TODO: maximize variance while minimize difference in aug graph
# -> maximize variance and bound the cosine similarity lower bound of aug_Z and Z
# TODO: freeze data augmenter while (minimized avg reconstruction loss / min KL divergence of edge prob)
# TODO: Also minimize aug loss variance for encoder
# TODO 1: Siamese Network augmented view doesn't backpropagate (only provide constrastive term)
# TODO 2: debiased CL (clustering)
# TODO: reduce number of channels
# TODO: activation / dropout
# TODO: linear layer++ batch normalization
# TODO: cosine similarity weight
# TODO: check why loss is small in train encoder
# TODO: dont consider CL in augmenter
# TODO: use APPNP output for CL
# TODO: use original cosine sim for CL, not sigmoid prob -> poor performance
# TODO: try to move intra-view CL loss out of loss function, aug graph does not consider intra-view CL loss -> poor performance
# TODO: bias_Zs in generation as ans for Bias_Zs in encoder output
# TODO: use adding normal distribution^(bias_Z) as noise
# more layers, diff propagate, AWX, lower learning rate

# TODO: BYOL term of CL, barlow twins term
# TODO: CL divisor = 1 not pos pair / use cross entropy
# TODO: use MLP data augmenter to maximize loss but choose only a small fraction of edge as 1

# MLP version
# def generate_augmented_views(device, data_augmenter, data_augmenter_optimizer, Z_for_aug, adj_label, mean, logstd, train_mask, norm, weight_tensor, alpha, beta, gamma, temperature, T = 10):
#     # Train Data Augmenter for T epochs for maximizing loss variance among aug graphs, encoder shall be freeze
#     # torch.autograd.set_detect_anomaly(True)
#     data_augmenter.train()
#     data_augmenter.unfreeze()
#     # target_degree = torch.sum(adj_label.to_dense(), dim = 1)
#     for t in range(T):
#         data_augmenter_optimizer.zero_grad()

#         bias_Z = data_augmenter(Z_for_aug)
#         aug_graph = dot_product_decode(bias_Z)
        
#         aug_loss = loss_function(aug_graph, adj_label, mean, logstd, norm, weight_tensor, alpha, beta, train_mask)
#         intra_CL = intra_view_CL_loss(device, bias_Z, adj_label, gamma, temperature)
#         aug_loss = aug_loss + intra_CL

#         # Z_norm = torch.norm(Z_for_aug, p = 2, dim = 1).unsqueeze(-1)
#         # Z_norm = torch.where(Z_norm == 0., torch.ones_like(Z_norm), Z_norm)
#         # norm_Z = torch.div(Z_for_aug, Z_norm)
#         # norm_bias = torch.norm(bias_Zs[i], p = 2, dim = 1).unsqueeze(-1)
#         # norm_bias = torch.where(norm_bias == 0., torch.ones_like(norm_bias), norm_bias)
#         # norm_bias_Z = torch.div(bias_Zs[i], norm_bias)
#         # cos_sims[i] = torch.mean(torch.sum((norm_Z * norm_bias_Z), dim = 1))

#         # From Degree Persepctive
#         # degrees_diff = F.l1_loss(torch.sum(aug_graph, dim = 1), target_degree) # degrees_diff += F.mse_loss(torch.sum(aug_graphs[i], dim = 1), target_degree)
#         diff = F.mse_loss(aug_graph, adj_label.to_dense())
        
#         ###***###
#         loss = -1 * aug_loss
#         # loss = -1 * diff
#         print(loss)
#         loss.backward()
#         data_augmenter_optimizer.step()
    
#     data_augmenter.eval()
#     data_augmenter.freeze()
#     with torch.no_grad():
#         bias_Z = data_augmenter(Z_for_aug)

#     # Rebuild Adjacency Matrix from bias_Z
#     aug_graph = dot_product_decode(bias_Z.detach())
#     mask = torch.zeros(aug_graph.shape[0], aug_graph.shape[0], dtype = torch.bool, requires_grad = False, device = bias_Z.device)
#     values, indices = torch.topk((torch.triu(aug_graph)-torch.eye(aug_graph.shape[0], device = bias_Z.device)).flatten(), int((torch.triu(adj_label.to_dense())-torch.eye(aug_graph.shape[0], device = bias_Z.device)).sum()), largest = True)
#     mask.flatten()[indices] = True
#     mask = mask + mask.T + torch.eye(aug_graph.shape[0], dtype = torch.bool, device = bias_Z.device)
#     aug_graph[mask] = 1
#     aug_graph[~mask] = 0

#     return aug_graph.to_sparse().indices(), 0.0

# Another VGAE version
# def generate_augmented_views(device, data_augmenter, data_augmenter_optimizer, ori_repr, features, adj_norm, adj_label, train_mask, norm, weight_tensor, alpha, beta, gamma, temperature):
#     # Train Data Augmenter for T epochs for maximizing loss variance among aug graphs, encoder shall be freeze
#     # torch.autograd.set_detect_anomaly(True)
#     data_augmenter.train()
#     # target_degree = torch.sum(adj_label.to_dense(), dim = 1)
#     for t in range(10):
#         data_augmenter_optimizer.zero_grad()

#         Z = data_augmenter(features, adj_norm)
#         graph = dot_product_decode(Z)
        
#         aug_loss = loss_function(graph, adj_label, data_augmenter.mean, data_augmenter.logstd, norm, weight_tensor, alpha, 0, train_mask)
#         # intra_CL = intra_view_CL_loss(device, Z, adj_label, gamma, temperature)
#         # aug_loss = aug_loss + intra_CL
#         print('aug_loss', aug_loss)
        
#         ## cos_sim
#         # ori_repr_norm = torch.norm(ori_repr, p = 2, dim = 1).unsqueeze(-1)
#         # ori_repr_norm = torch.where(ori_repr_norm == 0., torch.ones_like(ori_repr_norm), ori_repr_norm)
#         # norm_ori_repr = torch.div(ori_repr, ori_repr_norm)
#         # Z_norm = torch.norm(Z, p = 2, dim = 1).unsqueeze(-1)
#         # Z_norm = torch.where(Z_norm == 0., torch.ones_like(Z_norm), Z_norm)
#         # norm_Z = torch.div(Z, Z_norm)
#         # cos_sims = torch.mean(torch.sum((norm_Z * norm_ori_repr), dim = 1))
#         # print(cos_sims)
#         ##

#         ## BYOL
#         exp_cos_sim = torch.exp(torch.sigmoid(torch.matmul(Z.t(), ori_repr)))
#         pos = torch.diag(exp_cos_sim)
#         neg = torch.sum(exp_cos_sim, dim = 1)
#         neg = torch.where(neg == 0., torch.ones_like(neg), neg)
#         CL_Loss = -1.0 * gamma * torch.mean(torch.log(torch.div(pos, neg)))
#         print('CL_Loss', CL_Loss)
#         ##

#         ## CL
#         # exp_cos_sim = torch.exp(torch.sigmoid(torch.matmul(Z, ori_repr.t())))
#         # pos = torch.diag(exp_cos_sim)
#         # neg = torch.sum(exp_cos_sim, dim = 1)
#         # neg = torch.where(neg == 0., torch.ones_like(neg), neg)
#         # CL_Loss = -1.0 * gamma * torch.mean(torch.log(torch.div(pos, neg)))
#         # print(CL_Loss)
#         ##

#         # From Degree Persepctive
#         # degrees_diff = F.l1_loss(torch.sum(aug_graph, dim = 1), target_degree) # degrees_diff += F.mse_loss(torch.sum(aug_graphs[i], dim = 1), target_degree)
#         # diff = F.mse_loss(aug_graph, adj_label.to_dense())
        
#         ###***###
#         loss = CL_Loss / aug_loss
#         print('loss', loss)
#         loss.backward()
#         data_augmenter_optimizer.step()
    
#     data_augmenter.eval()
#     with torch.no_grad():
#         bias_Z = data_augmenter(features, adj_norm)

#     # Rebuild Adjacency Matrix from bias_Z
#     aug_graph = dot_product_decode(bias_Z.detach())
#     mask = torch.zeros(aug_graph.shape[0], aug_graph.shape[0], dtype = torch.bool, requires_grad = False, device = bias_Z.device)
#     values, indices = torch.topk((torch.triu(aug_graph)-torch.eye(aug_graph.shape[0], device = bias_Z.device)).flatten(), int((torch.triu(adj_label.to_dense())-torch.eye(aug_graph.shape[0], device = bias_Z.device)).sum()), largest = True)
#     mask.flatten()[indices] = True
#     mask = mask + mask.T + torch.eye(aug_graph.shape[0], dtype = torch.bool, device = bias_Z.device)
#     aug_graph[mask] = 1
#     aug_graph[~mask] = 0

#     return aug_graph.to_sparse().indices(), 0.0

def train_encoder(dataset_str, device, num_epoch, adj, features, hidden1, hidden2, dropout, learning_rate, weight_decay, aug_graph_weight, aug_ratio, aug_bound, alpha, beta, gamma, delta, temperature, labels, idx_train, idx_val, idx_test, ver, degree_threshold, loss_ver):
    num_nodes = adj.shape[0]
    # nb_classes = labels.shape[1]
    
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    if dataset_str in ['ogbl-ddi','ogbl-collab']:
        print("ogbl!!")
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_ogbl(adj,dataset_str, idx_train, idx_val, idx_test)
        print(f"{type(adj_train)}, {type(train_edges)}, {type(val_edges)}, {type(val_edges_false)}, {type(test_edges)}, {type(test_edges_false)}")
    else:
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj,dataset_str)
        print(f"{type(adj_train)}, {type(train_edges)}, {type(val_edges)}, {type(val_edges_false)}, {type(test_edges)}, {type(test_edges_false)}")
         
    adj = adj_train
    train_mask = torch.ones(num_nodes*num_nodes, dtype = torch.bool, requires_grad = False).to(device)
    for r, c in val_edges:
        train_mask[num_nodes * r + c] = False
    for r, c in test_edges:
        train_mask[num_nodes * r + c] = False
    training_instance_number = torch.sum(train_mask).item()

    # APPNP
    edge_index = from_scipy_sparse_matrix(adj)[0].to(device)
    ##
    if dataset_str in ['USAir', 'PB', 'Celegans', 'Power', 'Router', 'Ecoli', 'Yeast', 'NS','obgl-ddi']:
        print('Training Data Without Init Attr ...')
        # n2v
        features = CalN2V(edge_index, 16, 1)
        features = sp.lil_matrix(features.numpy())

        # ones
        # features = torch.ones(num_nodes, 512)
        # features = features.float()
        # features = sp.lil_matrix(features.numpy())

        # zeros
        # features = torch.zeros(num_nodes, 512)
        # features = features.float()
        # features = sp.lil_matrix(features.numpy())

        # onehot
        # features = F.one_hot(torch.arange(num_nodes), num_classes = num_nodes)
        # features = features.float()
        # features = sp.lil_matrix(features.numpy())

        # adj
        # features = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((adj.tocoo().row, adj.tocoo().col))), torch.FloatTensor(adj.data), torch.Size(adj.shape)).to_dense()
        # features = features.float()
        # features = sp.lil_matrix(features.numpy())

        # adj + eye
        # adj = adj + sp.eye(adj.shape[0])
        # features = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((adj.tocoo().row, adj.tocoo().col))), torch.FloatTensor(adj.data), torch.Size(adj.shape)).to_dense()
        # features = features.float()
        # features = sp.lil_matrix(features.numpy())

        # Laplacian Matrix
        # adj_norm = preprocess_graph(adj) 
        # features = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])).to_dense()
        # features = features.float()
        # features = sp.lil_matrix(features.numpy())

        # common neighbors
        # adj = adj + sp.eye(adj.shape[0])
        # g = nx.from_scipy_sparse_matrix(adj)
        # features = []
        # for i in range(g.number_of_nodes()):
        #     feat_i = []
        #     for j in range(g.number_of_nodes()):
        #         n = len(list(nx.common_neighbors(g, i, j)))
        #         feat_i.append(n)
        #     features.append(feat_i)
        # features = np.array(features)
        # features = sp.lil_matrix(features)
        # features = adj.multiply(features)
    ##
    feat_dim = features.shape[1]
    print(f'Node Nums: {num_nodes}, Init Feature Dim: {feat_dim}')
    
    #### ATTEMPT: use distance for decoder
    # nx_graph = nx.from_scipy_sparse_matrix(adj_train + sp.eye(adj_train.shape[0]))
    # all_pair_path_length = dict(nx.all_pairs_shortest_path_length(nx_graph))
    # distance_weight = torch.ones([num_nodes, num_nodes])
    # for i in range(num_nodes):
    #     max_dist_to_i = max(all_pair_path_length[i].values())
    #     if max_dist_to_i == 0:
    #         continue
    #     for j in range(num_nodes):
    #         if j in all_pair_path_length[i]:
    #             distance_weight[i][j] = all_pair_path_length[i][j] / max_dist_to_i
    
    #     distance_weight[i][i] = 1
    #     min_weight = torch.min(distance_weight[i])
    #     distance_weight[i][i] = min_weight
    # distance_weight = distance_weight.to(device)
    ####

    # Some preprocessing
    adj_norm = preprocess_graph(adj) #Laplacian Matrix
    # _, features = preprocess_features(features)
    features = sparse_to_tuple(features.tocoo())
    
    # Create Model
    pos_weight = float(training_instance_number - adj.sum()) / adj.sum() # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = training_instance_number / float((training_instance_number - adj.sum()) * 2) # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])).to(device)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2])).to(device)
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2])).to(device)
    weight_mask = adj_label.to_dense().view(-1)[train_mask] == 1 # weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    # init model and optimizer
    encoder = VGNAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device) # encoder = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    optimizer = Adam(encoder.parameters(), lr = learning_rate, weight_decay = weight_decay)

    data_augmenter = MLP(hidden2, hidden2).to(device)
    # data_augmenter = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    data_augmenter_optimizer = Adam(data_augmenter.parameters(), lr = 0.01, weight_decay = weight_decay)

    best_acc = 0.0; best_roc = 0.0; best_ap = 0.0
    best_hit1 = 0.0;best_hit3 = 0.0;best_hit10 = 0.0; best_hit20 = 0.0; best_hit50 = 0.0; 
    best_hit1_roc = 0.0;best_hit3_roc = 0.0;best_hit10_roc = 0.0; best_hit20_roc = 0.0; best_hit50_roc = 0.0; 
    best_hit1_ep = 0.0;best_hit3_ep = 0.0;best_hit10_ep = 0.0; best_hit20_ep = 0.0; best_hit50_ep = 0.0; 
    modification_ratio_history = []
    roc_history = []
    # train model

    neighbors = {}

    for u, v in zip(edge_index[0], edge_index[1]):
        if u not in neighbors:
            neighbors[u] = set()
        if v not in neighbors:
            neighbors[v] = set()
        neighbors[u].add(v)
        neighbors[v].add(u)

    # common_neighbors_count = {}
    common_neighbors_count = np.zeros((num_nodes, num_nodes), dtype=int)
    total = 0
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):  
            if u in neighbors and v in neighbors:
                common_neighbors = neighbors[u].intersection(neighbors[v])
                common_neighbors_count[u][v] = common_neighbors_count[v][u] = len(common_neighbors)
                total += len(common_neighbors)
    avg_cn_cnt = float(total) / (num_nodes*(num_nodes-1)/2)
    
    feat_sim = cosine_similarity(features.to_dense().cpu())
    # for u in neighbors.keys():
    #     for v in neighbors.keys():
    #         if u < v:  
    #             if u in neighbors and v in neighbors:
    #                 common_neighbors = neighbors[u].intersection(neighbors[v])               
    #                 common_neighbors_count[(u, v)] = len(common_neighbors)
    #                 common_neighbors_count[(v, u)] = len(common_neighbors)
    degree = np.array(adj_label.to_dense().cpu().sum(0)).squeeze()
    for epoch in tqdm(range(num_epoch)):
        t = time.time()
        encoder.train()
        optimizer.zero_grad()
        
        Z = encoder(features, edge_index) # Z = encoder(features, adj_norm)
        hidden_repr = encoder.Z

        # original loss
        A_pred = dot_product_decode(Z)

                # adjusted_weight_tensor = weight_tensor * torch.abs(A_pred.view(-1)[train_mask] - adj_label.to_dense().view(-1)[train_mask]).detach()
                # recon_loss = loss_function(A_pred, adj_label, encoder.mean, encoder.logstd, norm, adjusted_weight_tensor, alpha, beta, train_mask)
        recon_loss = loss_function(A_pred, adj_label, encoder.mean, encoder.logstd, norm, weight_tensor, alpha, beta, train_mask)
        if(loss_ver=="nei"):
            ori_intra_CL = inter_view_CL_loss(device, Z, Z, adj_label, gamma, temperature)
        else:
            ori_intra_CL = intra_view_CL_loss(device, Z, adj_label, gamma, temperature)
        loss = recon_loss + ori_intra_CL
        
        # Generate K graphs
        if epoch % 10 == 0:
            if epoch != 0:
                del aug_edge_index # del aug_edge_weights # del aug_adj_labels # del aug_norms # del aug_weight_tensors
                torch.cuda.empty_cache()
            print(f'loss: {loss}, recon loss: {recon_loss}')

            ###
            k = (num_nodes-1) * num_nodes * aug_ratio
            if(ver=="origin"):
                g, modification_ratio = Graph_Modify_Constraint(Z.detach(), adj_label.to_dense(), int(k), aug_bound)
            if(ver=="thm_exp"):
                g, modification_ratio = Graph_Modify_Constraint_exp(Z.detach(), adj_label.to_dense(), int(k), aug_bound)
            if(ver=="random"):
                g, modification_ratio = aug_random_edge(adj_label.to_dense(),aug_ratio)
            if(ver=="local"):
                g, modification_ratio = Graph_Modify_Constraint_local(Z.detach(), adj_label.to_dense(), int(k), aug_bound, common_neighbors_count, avg_cn_cnt)
            if(ver=="feat"):
                g, modification_ratio = Graph_Modify_Constraint_feat(Z.detach(), adj_label.to_dense(), int(k), aug_bound, feat_sim)
            if(ver=="uncover"):
                g, modification_ratio = degree_aug(Z.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)
                
            ## random modification
            # g, modification_ratio = aug_random_edge(adj_label.to_dense(),aug_ratio)
                
            aug_edge_index = g.to_sparse().indices()
            print(edge_index.shape)
            print(aug_edge_index.shape)
            # aug_edge_index, modification_ratio = generate_augmented_views(device, data_augmenter, data_augmenter_optimizer, encoder.Z.clone().detach(), adj_label, encoder.mean.clone().detach(), encoder.logstd.clone().detach(), train_mask, norm, weight_tensor, alpha, beta, gamma, temperature)
            # aug_edge_index, modification_ratio = generate_augmented_views(device, data_augmenter, data_augmenter_optimizer, hidden_repr.detach(), features, adj_norm, adj_label, train_mask, norm, weight_tensor, alpha, beta, gamma, temperature)
            ###

        modification_ratio_history.append(modification_ratio)

        # Calcualte Augment View
        bias_Z = encoder(features, aug_edge_index)

        # Overall losses        
        loss += inter_view_CL_loss(device, hidden_repr, encoder.Z.detach(), adj_label, delta, temperature)
        aug_loss = loss_function(dot_product_decode(bias_Z), adj_label, encoder.mean, encoder.logstd, norm, weight_tensor, alpha, beta, train_mask) # aug_loss = loss_function(dot_product_decode(bias_Z), aug_adj_labels[i], encoder.mean, encoder.logstd, aug_norms[i], aug_weight_tensors[i], alpha, beta, train_mask)
        if(loss_ver=="nei"):
            intra_CL = inter_view_CL_loss(device, bias_Z, bias_Z, adj_label, gamma, temperature)
        else:
            intra_CL = intra_view_CL_loss(device, bias_Z, adj_label, gamma, temperature)
        aug_losses = aug_loss  + intra_CL
        loss += aug_losses * aug_graph_weight
        #print(f'aug_loss: {aug_loss}, intra_CL: {intra_CL}')

        # Update Model
        loss.backward()
        optimizer.step()
        
        del bias_Z
        del encoder.Z
        del encoder.mean
        del encoder.logstd
        del Z
        del hidden_repr
        torch.cuda.empty_cache()
        ########################################################
        # Evaluate edge prediction
        encoder.eval()
        with torch.no_grad():
            Z = encoder(features, edge_index) # Z = encoder(features, adj_norm)
            A_pred = dot_product_decode(Z)
        # A_pred = train_decoder(device, encoder.Z.clone().detach(), adj_label, weight_tensor, norm, train_mask)
        print(A_pred.shape)
        train_acc = get_acc(A_pred.data.cpu(), adj_label.data.cpu())
        val_roc, val_ap, val_hit = get_scores(dataset_str, val_edges, val_edges_false, A_pred.data.cpu().numpy(), adj_orig)
        test_roc, test_ap, test_hit = get_scores(dataset_str, test_edges, test_edges_false, A_pred.data.cpu().numpy(), adj_orig)
        roc_history.append(test_roc)
        # val_acc, test_acc = logist_regressor_classification(device = device, Z = encoder.Z.clone().detach(), labels = labels, idx_train = idx_train, idx_val = idx_val, idx_test = idx_test)
        print(f'Epoch: {epoch + 1}, train_loss= {loss.item():.4f}, train_acc= {train_acc:.4f}, val_roc= {val_roc:.4f}, val_ap= {val_ap:.4f}, test_roc= {test_roc:.4f}, test_ap= {test_ap:.4f}, time= {time.time() - t:.4f}')
        print(f'Hit@K for val: 1={val_hit[0]}, 3={val_hit[1]}, 10={val_hit[2]}, 20={val_hit[3]}')
        print(f'Hit@K for test: 1={test_hit[0]}, 3={test_hit[1]}, 10={test_hit[2]}, 20={test_hit[3]}')
        # print(f'Hit@K for test: 10={test_hit[0]}, 20={test_hit[1]}, 50={test_hit[2]}')
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     best_test_acc = test_acc
        #     best_classi_epoch = epoch
        #     print(f'Update Best Acc, Epoch = {epoch+1}, Val_acc = {best_acc:.3f}, Test_acc = {test_acc:.3f}')
        if test_hit[0] > best_hit1:
            best_hit1 = test_hit[0]
            best_hit1_roc = test_roc
            best_hit1_ep = epoch 
        if test_hit[1] > best_hit3:
            best_hit3 = test_hit[1]
            best_hit3_roc = test_roc
            best_hit3_ep = epoch 
        if test_hit[2] > best_hit10:
            best_hit10 = test_hit[2]
            best_hit10_roc = test_roc
            best_hit10_ep = epoch
        if test_hit[3] > best_hit20:
            best_hit20 = test_hit[3]
            best_hit20_roc = test_roc
            best_hit20_ep = epoch    
        if val_roc > best_roc:
            best_roc = val_roc
            best_test_roc = test_roc
            best_ap = val_ap
            best_test_ap = test_ap
            best_link_epoch = epoch
            print(f'Update Best Roc, Epoch = {epoch+1}, Val_roc = {best_roc:.3f}, val_ap = {best_ap:.3f}, test_roc = {best_test_roc:.3f}, test_ap = {best_test_ap:.3f}')
        print('-' * 100)

    # print(f'best classification epoch = {best_classi_epoch+1}, val_acc = {best_acc:.3f}, test_acc = {best_test_acc:.3f}')
    print(f'best link prediction epoch = {best_link_epoch+1}, Val_roc = {best_roc:.3f}, val_ap = {best_ap:.3f}, test_roc = {best_test_roc:.3f}, test_ap = {best_test_ap:.3f}')
    print(f'best hit@1 epoch = {best_hit1_ep}, hit@1 = {best_hit1}, val = {best_hit1_roc}')
    print(f'best hit@3 epoch = {best_hit3_ep}, hit@3 = {best_hit3}, val = {best_hit3_roc}')
    print(f'best hit@10 epoch = {best_hit10_ep}, hit@10 = {best_hit10}, val = {best_hit10_roc}')
    print(f'best hit@20 epoch = {best_hit20_ep}, hit@20 = {best_hit20}, val = {best_hit20_roc}')
    # print(f'best hit@50 epoch = {best_hit50_ep}, hit@50 = {best_hit50}, val = {best_hit50_roc}')
    test_roc, test_ap, test_hit = get_scores(dataset_str,test_edges, test_edges_false, A_pred.data.cpu(), adj_orig)
    print(f'End of training!\ntest_roc={test_roc:.5f}, test_ap={test_ap:.5f}')
    print(f'Hit@K for test: 1={test_hit[0]}, 3={test_hit[1]}, 10={test_hit[2]}, 20={test_hit[3]}')
    
    return encoder.Z.clone().detach(), roc_history, modification_ratio_history, edge_index



def get_scores(dataset_str,edges_pos, edges_neg, adj_rec, adj_orig):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Predict on test set of edges
    preds = []
    pos = []
    
    pos_for_hitsk = []
    neg_for_hitsk = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        # print(sigmoid(adj_rec[e[0], e[1]].item()))
        preds.append(adj_rec[e[0], e[1]].item())
        pos.append(adj_orig[e[0], e[1]])
        
        
        # pos_for_hitsk.append(adj_rec[e[0], e[1]].item())
        
        
    preds_neg = []
    neg = []
    for e in edges_neg:
        # preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        preds_neg.append(adj_rec[e[0], e[1]].item())
        neg.append(adj_orig[e[0], e[1]])
        # neg_for_hitsk.append(adj_rec[e[0], e[1]].item())
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    hitk = []
    # evaluator = Evaluator(name=dataset_str)
    for k in [1,3,10,20]:
        hitk.append(eval_hits(torch.tensor(preds), torch.tensor(preds_neg),k))
        # evaluator.K = k
        # hits = evaluator.eval({
            # 'y_pred_pos': torch.tensor(preds),
            # 'y_pred_neg': torch.tensor(preds_neg),
        # })[f'hits@{k}']
        # hitk.append(hits)
    return roc_score, ap_score, hitk

def eval_hits(y_pred_pos, y_pred_neg, K):
    '''
    compute Hits@K
    For each positive target node, the negative target nodes are the same.
    y_pred_neg is an array.
    rank y_pred_pos[i] against y_pred_neg for each i
    From:
    https://github.com/snap-stanford/ogb/blob/1c875697fdb20ab452b2c11cf8bfa2c0e88b5ad3/ogb/linkproppred/evaluate.py#L214
    '''

    if len(y_pred_neg) < K:
        print(f'[WARNING]: hits@{K} defaulted to 1')
        return 1.0 #{'hits@{}'.format(K): 1.0}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K,largest=True)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(
        y_pred_pos
    )
    return hitsK
    # return {'hits@{}'.format(K): hitsK}

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# def train_decoder(device, Z, adj_label, weight_tensor, norm, train_mask):
#     num_nodes = Z.shape[0]
#     feat_dim = Z.shape[1]
#     decoder = Decoder(feat_dim, feat_dim).to(device)
#     opt = Adam(decoder.parameters(), lr = 0.01, weight_decay = 0.0)

#     for _ in range(100):
#         A_pred = decoder(Z)
#         loss = norm * F.binary_cross_entropy(A_pred.view(-1)[train_mask], adj_label.to_dense().view(-1)[train_mask], weight = weight_tensor)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
    
#     with torch.no_grad():
#         A_pred = decoder(Z).detach()
#     del decoder
#     del opt
#     return A_pred

def train_classifier(device, Z, labels, idx_train, idx_val, idx_test):
    hid_units = Z.shape[1]
    nb_classes = labels.shape[1]

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    
    Z = torch.FloatTensor(normalize(Z.cpu().numpy(), norm='l2')).to(device)
    
    train_embs = Z[idx_train].detach()
    val_embs = Z[idx_val].detach()
    test_embs = Z[idx_test].detach()

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    xent = nn.CrossEntropyLoss()
    tot = torch.zeros(1).to(device)
    accs = []
    for _ in range(50):
        log = LogReg(hid_units, nb_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr = 0.01, weight_decay = 0.0)

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            loss.backward()
            opt.step()
        
        log.eval()
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        # print('acc:[{:.4f}]'.format(acc))
        tot += acc
    
    print('-' * 100)
    print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
    accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))
    print('-' * 100)

def logist_regressor_classification(device, Z, labels, idx_train, idx_val, idx_test):
    hid_units = Z.shape[1]
    nb_classes = labels.shape[1]

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    
    Z = torch.FloatTensor(normalize(Z.cpu().numpy(), norm='l2')).to(device)
    
    train_embs = Z[idx_train].detach().data.cpu()
    val_embs = Z[idx_val].detach().data.cpu()
    test_embs = Z[idx_test].detach().data.cpu()

    train_lbls = torch.argmax(labels[0, idx_train], dim=1).detach().data.cpu()
    val_lbls = torch.argmax(labels[0, idx_val], dim=1).detach().data.cpu()
    test_lbls = torch.argmax(labels[0, idx_test], dim=1).detach().data.cpu()

    tot = torch.zeros(1)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c), n_jobs=8, cv=5, verbose=0)

    clf.fit(train_embs, train_lbls)
    
    # val
    logits = clf.predict_proba(val_embs)
    preds = torch.argmax(torch.tensor(logits), dim=1)
    val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
    print('val_acc:[{:.4f}]'.format(val_acc))

    # test
    logits = clf.predict_proba(test_embs)
    preds = torch.argmax(torch.tensor(logits), dim=1)
    test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    print('test_acc:[{:.4f}]'.format(test_acc))

    return val_acc, test_acc
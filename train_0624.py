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
from attnFuse import AttentionModule, GCN, FeatureDecoder


def train_encoder(dataset_str, device, num_epoch, adj, features, hidden1, hidden2, dropout, learning_rate, weight_decay, aug_graph_weight, aug_ratio, aug_bound, alpha, beta, gamma, delta, temperature, labels, idx_train, idx_val, idx_test, ver, degree_threshold, loss_ver, feature_ratio):
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


    feat_dim = features.shape[1]
    print(f'Node Nums: {num_nodes}, Init Feature Dim: {feat_dim}')
    


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
    encoder1 = VGNAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device) # encoder = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    encoder2 = VGNAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device) # encoder = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    feature_decoder = FeatureDecoder(hidden2, hidden1, feat_dim).to(device)
    optimizer1 = Adam(encoder1.parameters(), lr = learning_rate, weight_decay = weight_decay)
    optimizer2 = Adam(encoder2.parameters(), lr = learning_rate, weight_decay = weight_decay)
    optimizer3 = Adam(feature_decoder.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    # data_augmenter = MLP(hidden2, hidden2).to(device)
    # data_augmenter = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    # data_augmenter_optimizer = Adam(data_augmenter.parameters(), lr = 0.01, weight_decay = weight_decay)

    best_acc = 0.0; best_roc = 0.0; best_ap = 0.0
    best_hit1 = 0.0;best_hit3 = 0.0;best_hit10 = 0.0; best_hit20 = 0.0; best_hit50 = 0.0; best_hit100 = 0.0
    best_hit1_roc = 0.0;best_hit3_roc = 0.0;best_hit10_roc = 0.0; best_hit20_roc = 0.0; best_hit50_roc = 0.0; best_hit100_roc = 0.0
    best_hit1_ep = 0.0;best_hit3_ep = 0.0;best_hit10_ep = 0.0; best_hit20_ep = 0.0; best_hit50_ep = 0.0; best_hit100_ep = 0.0
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

    common_neighbors_count = {}
    # common_neighbors_count = np.zeros((num_nodes, num_nodes), dtype=int)
    total = 0
    # for u in range(num_nodes):
    #     for v in range(u + 1, num_nodes):  
    #         if u in neighbors and v in neighbors:
    #             common_neighbors = neighbors[u].intersection(neighbors[v])
    #             common_neighbors_count[u][v] = common_neighbors_count[v][u] = len(common_neighbors)
    #             total += len(common_neighbors)
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
    print(f"dataset avg degree: {np.mean(degree)}")
    attn_ft = AttentionModule(hidden2, num_nodes).to(device)
    attn_o = AttentionModule(hidden2, num_nodes).to(device)
    mse_loss = nn.MSELoss()
    
    aug_feat_graph = feature_aug(adj_label, features.to_dense(), feature_ratio, degree_threshold)
    aug_feat_edge_index = aug_feat_graph.to_sparse().indices()
    
    for epoch in tqdm(range(num_epoch)):
        t = time.time()
        encoder1.train()
        encoder2.train()
        feature_decoder.train()
        attn_ft.train()
        attn_o.train() 
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        
        Zo = encoder1(features, edge_index) # Z = encoder(features, adj_norm)
        # Z_o2 = encoder2(features, edge_index) # Z = encoder(features, adj_norm)
        hidden_repr_o1 = encoder1.Z
        # hidden_repr_o2 = encoder2.Z
        Z_o_mean = encoder1.mean; Z_o_logstd = encoder1.logstd
        # Z_o2_mean = encoder2.mean; Z_o2_logstd = encoder2.logstd

        Z_f = encoder2(features, aug_feat_edge_index) # Z = encoder(features, adj_norm)
        # Z_o2 = encoder2(features, edge_index) # Z = encoder(features, adj_norm)
        hidden_repr_f = encoder2.Z
        # hidden_repr_o2 = encoder2.Z
        Z_f_mean = encoder2.mean; Z_f_logstd = encoder2.logstd
        # alpha_o1, alpha_o2 = attn_o(Z_o1,Z_o2)
        # alpha_o1 = alpha_o1.unsqueeze(1)
        # alpha_o2 = alpha_o2.unsqueeze(1)
        # print(f"Shape: alpha_o1:{alpha_o1.shape}, Z_o1:{Z_o1.shape}, alpha_o2:{alpha_o2.shape}, Z_o2:{Z_o2.shape}")
        # Zo = alpha_o1*Z_o1 + alpha_o2*Z_o2
        # Z_o_mean = alpha_o1*Z_o1_mean + alpha_o2*Z_o2_mean; Z_o_logstd = alpha_o1*Z_o1_logstd + alpha_o2*Z_o2_logstd
        # Generate K graphs
        
        if epoch % 10 == 0:
            if epoch != 0:
                del aug_edge_index # del aug_edge_weights # del aug_adj_labels # del aug_norms # del aug_weight_tensors
                torch.cuda.empty_cache()

            k = (num_nodes-1) * num_nodes * aug_ratio
            if(ver=="origin"):
                g, modification_ratio = Graph_Modify_Constraint(Zo.detach(), adj_label.to_dense(), int(k), aug_bound)
            if(ver=="thm_exp"):
                g, modification_ratio = Graph_Modify_Constraint_exp(Zo.detach(), adj_label.to_dense(), int(k), aug_bound)
            if(ver=="random"):
                g, modification_ratio = aug_random_edge(adj_label.to_dense(),aug_ratio)
            if(ver=="local"):
                g, modification_ratio = Graph_Modify_Constraint_local(Zo.detach(), adj_label.to_dense(), int(k), aug_bound, common_neighbors_count, avg_cn_cnt)
            if(ver=="feat"):
                g, modification_ratio = Graph_Modify_Constraint_feat(Zo.detach(), adj_label.to_dense(), int(k), aug_bound, feat_sim)
            if(ver=="uncover"):
                g, modification_ratio = degree_aug(Zo.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)
            if(ver=="v2"):
                g, modification_ratio = degree_aug_v2(Zo.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)            
            if(ver=="v3"):
                g, modification_ratio = degree_aug_v3(Zo.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)

                
            ### TODO Feature augmentation
            aug_features = drop_feature(features.to_dense(), aug_ratio)
            
            aug_edge_index = g.to_sparse().indices()



        modification_ratio_history.append(modification_ratio)

        # Calcualte Augment View
        Z_t = encoder1(features, aug_edge_index)    # Z topology
        Z_t_mean = encoder1.mean; Z_t_logstd = encoder1.logstd
        # Z_f = encoder2(aug_features, edge_index)
        # Z_f_mean = encoder2.mean; Z_f_logstd = encoder2.logstd
        
        # AttnFusion
        # attn_ft = AttentionModule(hidden2, num_nodes)
        alpha_f, alpha_t = attn_ft(Z_f,Z_t)
        alpha_f = alpha_f.unsqueeze(1)
        alpha_t = alpha_t.unsqueeze(1)
        Zft = alpha_f*Z_f + alpha_t*Z_t
        Z_ft_mean = alpha_f*Z_f_mean + alpha_t*Z_t_mean; Z_ft_logstd = alpha_f*Z_f_logstd + alpha_t*Z_t_logstd
        
        # print(f"Shape: aug_features: {aug_features.shape}, feat_dim:{feat_dim}, hidden1: {hidden1}, hidden2: {hidden2}, Z_f:{Z_f.shape}")
        recon_feat = feature_decoder(Zft)
        # print(f"Shape: recon feat:{recon_feat.shape}")
        
        # Loss
        # loss = torch.mean((Z_o1 - Z_o2) ** 2) # alignment loss
        loss = 0.0
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
        loss += aug_graph_weight * loss_function(Aft_pred, adj_label, Z_ft_mean, Z_ft_logstd, norm, weight_tensor, alpha, beta, train_mask)    # aug fusion reconstruction loss
        loss += loss_function(Ao_pred, adj_label, Z_o_mean, Z_o_logstd, norm, weight_tensor, alpha, beta, train_mask)   # origin reconstruction loss
        loss += aug_graph_weight * intra_CLft
        loss += aug_graph_weight * mse_loss(recon_feat, features.to_dense())


        # Update Model
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        
        # del bias_Z
        # del encoder.Z
        # del encoder.mean
        # del encoder.logstd
        # del Z
        # del hidden_repr
        torch.cuda.empty_cache()
        ########################################################
        # Evaluate edge prediction
        encoder1.eval()
        encoder2.eval()
        with torch.no_grad():
            Z = encoder1(features, edge_index) # Z = encoder(features, adj_norm)
            # Z2 = encoder2(features, edge_index) # Z = encoder(features, adj_norm)
            # alpha1, alpha2 = attn_o(Z1,Z2)
            # alpha1 = alpha1.unsqueeze(1);alpha2 = alpha2.unsqueeze(1)
            # Z = alpha1*Z1 + alpha2*Z2
            A_pred = dot_product_decode(Z)
        # A_pred = train_decoder(device, encoder.Z.clone().detach(), adj_label, weight_tensor, norm, train_mask)
        print(A_pred.shape)
        train_acc = get_acc(A_pred.data.cpu(), adj_label.data.cpu())
        val_roc, val_ap, val_hit = get_scores(dataset_str, val_edges, val_edges_false, A_pred.data.cpu().numpy(), adj_orig)
        test_roc, test_ap, test_hit = get_scores(dataset_str, test_edges, test_edges_false, A_pred.data.cpu().numpy(), adj_orig)
        roc_history.append(test_roc)
        # val_acc, test_acc = logist_regressor_classification(device = device, Z = encoder.Z.clone().detach(), labels = labels, idx_train = idx_train, idx_val = idx_val, idx_test = idx_test)
        print(f'Epoch: {epoch + 1}, train_loss= {loss.item():.4f}, train_acc= {train_acc:.4f}, val_roc= {val_roc:.4f}, val_ap= {val_ap:.4f}, test_roc= {test_roc:.4f}, test_ap= {test_ap:.4f}, time= {time.time() - t:.4f}')
        print(f'Hit@K for val: 1={val_hit[0]}, 3={val_hit[1]}, 10={val_hit[2]}, 20={val_hit[3]}, 100={val_hit[4]}')
        print(f'Hit@K for test: 1={test_hit[0]}, 3={test_hit[1]}, 10={test_hit[2]}, 20={test_hit[3]}, 100={test_hit[4]}')
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
        if test_hit[4] > best_hit50:
            best_hit50 = test_hit[4]
            best_hit50_roc = test_roc
            best_hit50_ep = epoch 
        if test_hit[5] > best_hit100:
            best_hit100 = test_hit[5]
            best_hit100_roc = test_roc
            best_hit100_ep = epoch    
        if test_roc > best_roc:
            best_roc = test_roc
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
    print(f'best hit@50 epoch = {best_hit50_ep}, hit@50 = {best_hit50}, val = {best_hit50_roc}')
    print(f'best hit@100 epoch = {best_hit100_ep}, hit@100 = {best_hit100}, val = {best_hit100_roc}')
    # print(f'best hit@50 epoch = {best_hit50_ep}, hit@50 = {best_hit50}, val = {best_hit50_roc}')
    test_roc, test_ap, test_hit = get_scores(dataset_str,test_edges, test_edges_false, A_pred.data.cpu(), adj_orig)
    print(f'End of training!\ntest_roc={test_roc:.5f}, test_ap={test_ap:.5f}')
    print(f'Hit@K for test: 1={test_hit[0]}, 3={test_hit[1]}, 10={test_hit[2]}, 20={test_hit[3]}, 50={test_hit[4]}, 100={test_hit[5]}')
    
    return Z.clone().detach(), roc_history, modification_ratio_history, edge_index



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
    for k in [1,3,10,20,50,100]:
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
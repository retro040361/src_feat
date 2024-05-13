'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import sys
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, WebKB, Amazon, AttributedGraphDataset, WikiCS, PPI, CitationFull, IMDB, Twitch, LastFMAsia
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix,dense_to_sparse,is_undirected
from torch_geometric.nn import Node2Vec
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def CalN2V(edge_index, dim, in_out_param):
    modeljure = Node2Vec(edge_index, embedding_dim=dim, walk_length=20,
                     context_size=10, walks_per_node=1,
                     num_negative_samples=1, p=1, q=in_out_param, sparse=True).to(device)
    loader = modeljure.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(modeljure.parameters()), lr=0.01)
    modeljure.train()
    total_loss = 0
    print('___Calculating Node2Vec features___')
    for i in range(201):
        total_loss=0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = modeljure.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i%20 == 0:
            print(f'Setp: {i:03d} /200, Loss : {total_loss:.4f}')
    output=(modeljure.forward()).cpu().clone().detach()
    del modeljure
    del loader
    torch.cuda.empty_cache()
    return output

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        # load the data: x, tx, allx, graph
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/citation/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        
        test_idx_reorder = parse_index_file("data/citation/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # labels = np.argmax(labels, axis=1)

        idx_train = list(range(len(y)))
        idx_val = list(range(len(y), len(y)+500))
        idx_test = test_idx_range.tolist()
    
    elif dataset in ['chameleon', 'crocodile', 'squirrel']:
        if dataset in ['crocodile', 'squirrel']:
            data = WikipediaNetwork('data/', dataset)[0]
        elif dataset == 'chameleon':
            tmp_feature = WikipediaNetwork('data/', dataset, geom_gcn_preprocess = False)[0].x
            data = WikipediaNetwork('data/', dataset)[0]
            data.x = tmp_feature

        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:, 0]].tolist()

    elif dataset in ['cornell', 'texas', 'wisconsin']:
        data = WebKB('data/', dataset.capitalize())[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())

        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:, 0]].tolist()

    elif dataset in ['amazon_photo']:
        data = Amazon('data/amazon_photo', 'Photo')[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()

        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['Facebook']:
        data = AttributedGraphDataset('data/Facebook', dataset)[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = (data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(data.y.shape[1]):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['Flickr']:
        data = AttributedGraphDataset('data/Flickr', dataset)[0]
        data.x = data.x.to_dense()
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['WikiCS']:
        data = WikiCS('data/WikiCS')[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:]].tolist()

    elif dataset in ['PPI']:
        graph_index = 19 # There are total 20 graph to use 0~19
        data = PPI('data/PPI/'+str(graph_index))[graph_index]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = (data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(data.y.shape[1]):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()
    
    elif dataset in ['amazon_computers']:
        data = Amazon('data/amazon_computers', 'Computers')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()

        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['Cora_ML']:
        data = CitationFull('data/Cora_ML', 'Cora_ML')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()

        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['IMDB']:
        data = IMDB('data/IMDB')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:]].tolist()

    elif dataset in ['Twitch']:
        # DE: 9498 nodes
        # EN: 7126 nodes
        # ES: 4648 nodes
        # FR: 6551 nodes
        # PT: 1912 nodes
        # RU: 4385 nodes
        name = 'ES'
        data = Twitch('data/Twitch', name)[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())

        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['LastFMAsia']:
        data = LastFMAsia('data/LastFMAsia')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())

        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()


    elif dataset in ['USAir', 'PB', 'Celegans', 'Power', 'Router', 'Ecoli', 'Yeast', 'NS']:
        data_dir = 'data/wo_attr/' + dataset + '.mat'
        print('Load data from: '+ data_dir)
        import scipy.io as sio
        net = sio.loadmat(data_dir)
        edge_index, _ = from_scipy_sparse_matrix(net['net'])
        node_num = torch.max(edge_index) + 1
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        adj = to_scipy_sparse_matrix(edge_index)

        features = None
        labels = None
        idx_train = None
        idx_val = None
        idx_test = None

    else:
        print('Not Implemented')
        return None

    return adj, features, labels, idx_train, idx_val, idx_test

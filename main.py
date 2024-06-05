import argparse
import sys
import atexit
import torch
import numpy as np
from train import train_encoder, train_classifier, logist_regressor_classification
from input_data import load_data
from utils import Visualize, Plot, Visualize_with_edge, gaussion_KDE, vMF_KDE
from torch_geometric.utils.convert import from_scipy_sparse_matrix


parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--dataset-str', type=str, default = 'cora', help='type of dataset.')
parser.add_argument('--epochs', type=int, default = 700, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default = 256, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default = 64, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default = 0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default = 0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default = 5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--aug_graph_weight', type=float, default = 1.0, help='augmented graph weight')
parser.add_argument('--aug_ratio', type=float, default = 0.1, help='augmented ratio')
parser.add_argument('--aug_bound', type=float, default = 0.1, help='augmented edge bound')
parser.add_argument('--alpha', type=float, default = 1.0, help='Reconstruction Loss Weight')
parser.add_argument('--beta', type=float, default = 1.0, help='KL Divergence Weight')
parser.add_argument('--gamma', type=float, default = 1.0, help='Contrastive Loss Weight')
parser.add_argument('--delta', type=float, default = 1.0, help='Inter Contrastive Loss Weight')
parser.add_argument('--temperature', type=float, default = 1.0, help='Contrastive Temperature')
parser.add_argument('--logging', type=bool, default = False, help='logging')
parser.add_argument('--date', type=str, default = "0000", help='date')
parser.add_argument('--ver', type=str, default = "origin", help='modified version') # [origin, thm_exp, ]
parser.add_argument('--idx', type=str, default = "1", help='index') # [1,2,3,4,5]

args = parser.parse_args()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)

def main():
    print(f'Dataset: {args.dataset_str}')

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    if args.dataset_str == 'pubmed':
        device = torch.device('cpu')

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset_str)

    Z, roc_history, modification_ratio_history, edge_index = train_encoder(args.dataset_str, device, args.epochs, adj, features, args.hidden1, args.hidden2, args.dropout, args.lr, args.weight_decay, 
                            args.aug_graph_weight, args.aug_ratio, args.aug_bound, args.alpha, args.beta, args.gamma, args.delta, args.temperature,
                            labels, idx_train, idx_val, idx_test, args.ver)
    
    Plot(args.dataset_str, roc_history, modification_ratio_history)
    gaussion_KDE(args.dataset_str, Z)
    vMF_KDE(args.dataset_str, Z)
    
    if labels is not None:
        train_classifier(device, Z, labels, idx_train, idx_val, idx_test)
        logist_regressor_classification(device, Z, labels, idx_train, idx_val, idx_test)
        Visualize(args.dataset_str, Z, labels)
        # Visualize_with_edge(args.dataset_str, Z, labels, from_scipy_sparse_matrix(adj)[0])
        


if __name__ == '__main__':
    if args.logging == True:
        old_stdout = sys.stdout
        log_file = open(f'log/{args.date}/{args.dataset_str}_{args.ver}_{args.idx}.log',"w")
        sys.stdout = log_file
        
        set_random_seed(args.seed)
        main()

        sys.stdout = old_stdout
        log_file.close()
    else:
        set_random_seed(args.seed)
        main()
        
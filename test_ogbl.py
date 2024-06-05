import torch
from torch_geometric.nn import VGAE, GCNConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import scipy.sparse as sp
from tqdm import tqdm
import numpy as np
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Load data and evaluator
dataset = PygLinkPropPredDataset(name='ogbl-ddi', root='dataset/')
split_edge = dataset.get_edge_split()
graph=dataset[0]
evaluator = Evaluator(name='ogbl-ddi')
data = dataset[0]
data.x = torch.ones((graph['num_nodes'], 1))
# Model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_channels = 32
encoder = Encoder(dataset.num_features, out_channels).to(device)
model = VGAE(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model, optimizer, data, device,split_edge):
    model.train()
    optimizer.zero_grad()
    # print(split_edge['train']['edge'])
    z = model.encode(data.x.to(device), split_edge['train']['edge'].t().to(device))
    loss = model.recon_loss(z, split_edge['train']['edge'].t().to(device))
    loss += (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate model
@torch.no_grad()
def test(model, data, device, split_edge, evaluator):
    model.eval()
    z = model.encode(data.x.to(device), split_edge['train']['edge'].t().to(device))
    link_probs = model.decoder.forward_all(z)
    y_pred_pos = []
    y_pred_neg = []
    for e in split_edge['test']['edge']:
        y_pred_pos.append(link_probs[e[0],e[1]].item())
    for e in split_edge['test']['edge_neg']:
        y_pred_neg.append(link_probs[e[0],e[1]].item())
    evaluator.K = 20
    results = evaluator.eval({
        'y_pred_pos': np.array(y_pred_pos),
        'y_pred_neg': np.array(y_pred_neg),
    })
    # print(results)
    return results['hits@20']


best_hit = 0
# Training loop
for epoch in tqdm(range(1, 500)):
    loss = train(model, optimizer, data, device, split_edge)
    if epoch % 10 == 0:
        results = test(model, data, device, split_edge, evaluator)
        best_hit = max(best_hit, results)    
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        



# train(model, optimizer, data, device, split_edge)
# test(model, data, device, split_edge, evaluator)
results = test(model, data, device, split_edge, evaluator)
print('Best Hits@20:', best_hit)
print('Last Hits@20:', results)

import torch
from torch_geometric.nn import VGAE, GCNConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

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
evaluator = Evaluator(name='ogbl-ddi')
data = dataset[0]

# Model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_channels = 32
encoder = Encoder(dataset.num_features, out_channels).to(device)
model = VGAE(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x.to(device), split_edge['train']['edge'].to(device))
    loss = model.recon_loss(z, split_edge['train']['edge'].to(device))
    loss += (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Evaluate model
@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data.x.to(device), split_edge['train']['edge'].to(device))
    link_probs = model.decoder.forward_all(z)
    return evaluator.eval({
        'y_pred_pos': link_probs[split_edge['test']['edge'].t()],
        'y_pred_neg': link_probs[split_edge['test']['edge_neg'].t()],
        'y_true_pos': torch.ones(split_edge['test']['edge'].size(0)),
        'y_true_neg': torch.zeros(split_edge['test']['edge_neg'].size(0)),
    })

results = test()
print('Hits@20:', results['hits@20'])

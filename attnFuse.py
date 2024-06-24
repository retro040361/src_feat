import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class AttentionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionModule, self).__init__()
       
        self.W_z1 = nn.Parameter(torch.randn(output_dim, input_dim))
        self.W_z2 = nn.Parameter(torch.randn(output_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        self.q = nn.Parameter(torch.randn(output_dim, 1))

    def forward(self, z1, z2):
        """
        T and H1 should have dimensions (batch_size, input_dim)
        """
        
        omega_z1 = torch.matmul(self.q.T, torch.tanh(torch.matmul(self.W_z1, z1.t()) + self.b[:, None]))
        omega_z2 = torch.matmul(self.q.T, torch.tanh(torch.matmul(self.W_z2, z2.t()) + self.b[:, None]))
        
    
        # omega_z1 = omega_z1.squeeze()
        # omega_z2 = omega_z2.squeeze()

        # 应用Softmax
        # alpha_z1, alpha_z2 = F.softmax(torch.stack([omega_z1, omega_z2]), dim=0)
        alphas = F.softmax(torch.cat([omega_z1, omega_z2], dim=0), dim=0)
        alpha_z1, alpha_z2 = alphas[0], alphas[1]

        return alpha_z1, alpha_z2
    
#     # 假设输入维度和输出维度为示例
# input_dim = 10
# output_dim = 5

# # 创建模型实例
# att_module = AttentionModule(input_dim, output_dim)

# # 创建一些随机数据来模拟T和H1
# T = torch.randn(10, input_dim)
# H1 = torch.randn(10, input_dim)

# # 计算注意力权重
# alpha_T, alpha_H1 = att_module(T, H1)
# print("Alpha_T:", alpha_T)
# print("Alpha_H1:", alpha_H1)

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        
        return x
    
class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureDecoder, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 通过第一个全连接层后使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层，直接输出（假设最后一层不需要激活函数）
        x = self.fc2(x)
        return x
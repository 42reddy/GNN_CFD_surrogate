import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# Encoder
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# Processoe
class GNNProcessor(MessagePassing):
    def __init__(self, hidden_dim, num_layers=3, dropout=0.2):
        super().__init__(aggr='mean')
        self.layers = nn.ModuleList()
        self.dropout = dropout

        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = self.propagate(edge_index, x=x, mlp=layer)
        return x

    def message(self, x_i, x_j, mlp):
        return mlp(torch.cat([x_i, x_j], dim=1))


# Decoder
class GNNDecoder(nn.Module):
    def __init__(self, hidden_dim, out_channels, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


# Network
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers=3, dropout=0.2):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_dim, dropout)
        self.processor = GNNProcessor(hidden_dim, num_layers, dropout)
        self.decoder = GNNDecoder(hidden_dim, out_channels, dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        x = self.processor(x, edge_index)
        x = self.decoder(x)
        return x
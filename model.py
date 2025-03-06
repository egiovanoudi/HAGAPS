import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv


def _init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
    elif isinstance(m, ARMAConv):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.kaiming_uniform_(m.init_weight)
        nn.init.kaiming_uniform_(m.root_weight)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param, nonlinearity="leaky_relu")
            elif 'bias' in name:
                nn.init.zeros_(param)


class HAGAPS(nn.Module):
    def __init__(self, hidden_dim, max_len):
        super(HAGAPS, self).__init__()
        after_cnn_size = (((max_len - 12 + 1) // 3) - 12 + 1) // 4

        self.site_mpn = Site_MPN(hidden_dim, max_len)
        self.gene_mpn = Gene_MPN(hidden_dim)
        self.lstm = nn.LSTM(2*after_cnn_size*hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ffn = nn.Sequential(
            nn.Linear(2*hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Softmax(dim=0)
        )

        self.apply(_init_weights)

    def forward(self, data):
        N = []

        h = self.site_mpn(data)
        for i, _ in enumerate(h):
            rest_sites = torch.cat((h[:i], h[i + 1:]), dim=0)
            N.append(rest_sites.sum(dim=0))
        N = torch.stack(N)
        h = self.gene_mpn(h, N)
        h = torch.flatten(h, start_dim=1)
        h, _ = self.lstm(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, 0.1)
        h = self.ffn(h)
        return h.flatten()


class Site_MPN(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(Site_MPN, self).__init__()

        self.hidden_size = hidden_size
        self.max_len = max_len

        self.embedding = nn.Embedding(5, hidden_size, padding_idx=0)  # For A, U, C, G
        self.arma1 = ARMAConv(hidden_size, hidden_size, dropout=0.1)
        self.arma2 = ARMAConv(hidden_size, hidden_size, dropout=0.1)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=12),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=12),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=4),
        )

        self.apply(_init_weights)

    def forward(self, data):
        x, edge, weight = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x)
        x = self.arma1(x, edge, edge_weight=weight)
        x = self.arma2(x.float(), edge, edge_weight=weight)
        x = x.reshape(-1, self.max_len, self.hidden_size).float()
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        return x


class Gene_MPN(nn.Module):
    def __init__(self, hidden_dim):
        super(Gene_MPN, self).__init__()

        self.lstm = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True, bidirectional=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)


    def forward(self, h, N):
        h_g = torch.cat([h, N], dim=-1)
        h_g, _ = self.lstm(h_g)
        h_g = F.relu(h_g)
        a = torch.bmm(h.transpose(1, 2), h_g).squeeze(2)
        w = F.softmax(a, dim=1).unsqueeze(2)
        b = torch.sum(w * h.unsqueeze(1), dim=1, keepdim=False)
        h_a = torch.cat([h_g, b], dim=2)
        return h_a.squeeze(1)
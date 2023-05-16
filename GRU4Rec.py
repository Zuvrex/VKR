import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4Rec(nn.Module):
    def __init__(self, num_items, embed_size, num_blocks, d_ff, output_sz, dropout_prob):
        super(GRU4Rec, self).__init__()
        self.gru = nn.GRU(input_size=num_items, hidden_size=embed_size, num_layers=num_blocks, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(embed_size, d_ff)
        self.fc2 = nn.Linear(d_ff, output_sz)

    def forward(self, x, hid=None):
        x, hid = self.gru(x, hid)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x, hid

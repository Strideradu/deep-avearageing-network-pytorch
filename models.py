import torch
import torch.nn as nn
import torch.nn.functional as F

class DAN(nn.Module):

    def __init__(self,
                 n_embed=10000,
                 d_embed=300,
                 d_hidden = 256,
                 d_out = 2,
                 dp = 0.2,
                 embed_weight = None):

        super(DAN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embed = nn.Embedding(n_embed, d_embed)

        if embed_weight:
            # embed_weight = inputs.vocab.vectors
            self.embed.weight.data.copy_(embed_weight)

        self.dropout1 = nn.Dropout(dp)
        self.bn1 = nn.BatchNorm1d(d_embed)
        self.fc1 = nn.Linear(d_embed, d_hidden)
        self.dropout2 = nn.Dropout(dp)
        self.bn2 = nn.BatchNorm1d(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)


    def forward(self, batch):
        text = batch.text
        label = batch.label

        x = self.embed(text, label)

        x = x.mean(dim=1)

        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x
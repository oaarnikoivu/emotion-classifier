import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(x)
        x = self.attention(x).squeeze(2)
        alpha = F.softmax(x, dim=1).unsqueeze(1)
        return alpha


class AttentionBiLSTM(nn.Module):
    def __init__(self, bert, hidden_size, num_layers, dropout, fc_dropout,
                 embed_dropout, num_classes):
        super(AttentionBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.bert = bert
        embedding_dim = 768

        self.embed_dropout = nn.Dropout(embed_dropout)

        self.bilstm = nn.LSTM(embedding_dim,
                              hidden_size,
                              num_layers,
                              dropout=(0 if num_layers == 1 else dropout),
                              bidirectional=True,
                              batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_dropout = nn.Dropout(fc_dropout)

        self.attention = Attention(hidden_size)

    def forward(self, text):
        with torch.no_grad():
            x = self.bert(text)[0]

        x = self.embed_dropout(x)
        y, _ = self.bilstm(x)
        y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]
        alpha = self.attention(y)
        r = alpha.bmm(y).squeeze(1)
        h = torch.tanh(r)
        logits = self.fc(h)
        logits = self.fc_dropout(logits)
        return logits, alpha

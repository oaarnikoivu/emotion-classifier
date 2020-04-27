import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertModel
from args import args

bert = DistilBertModel.from_pretrained(args['distilbert_pretrained_model'])


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, H):
        M = torch.tanh(H)
        M = self.attention(M).squeeze(2)
        alpha = F.softmax(M, dim=1).unsqueeze(1)
        return alpha


class AttentionBiLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, fc_dropout,
                 emb_layer_dropout, num_classes):
        super(AttentionBiLSTM, self).__init__()

        self.hidden_size = hidden_size

        if args['use_glove']:
            embedding_dim = args['glove_embedding_dim']
            self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)
        else:
            self.bert = bert
            embedding_dim = args['bert_embedding_dim']

        # embedding layer dropout
        self.emb_layer_dropout = nn.Dropout(emb_layer_dropout)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size,
                            num_layers,
                            dropout=(0 if num_layers == 1 else dropout),
                            bidirectional=True,
                            batch_first=True)

        # penultimate layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_dropout = nn.Dropout(fc_dropout)

        self.attention = Attention(hidden_size)

    def forward(self, text):
        if args['use_glove']:
            embedded = self.embedding(text)
        else:
            with torch.no_grad():
                embedded = self.bert(text)[0]

        embedded = self.emb_layer_dropout(embedded)
        y, _ = self.lstm(embedded)
        y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]
        alpha = self.attention(y)
        r = alpha.bmm(y).squeeze(1)
        h = torch.tanh(r)
        logits = self.fc(h)
        logits = self.fc_dropout(logits)
        return logits, alpha

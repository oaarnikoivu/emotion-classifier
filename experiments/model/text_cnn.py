import torch
import torch.nn.functional as F
import torch.nn as nn


class BertCNN(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()

        self.bert = bert
        embedding_dim = 768
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=opt.num_filters,
                      kernel_size=(fs, embedding_dim)) for fs in opt.filter_sizes
        ])

        self.fc = nn.Linear(len(opt.filter_sizes) * opt.num_filters, opt.output_dim)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]

        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)



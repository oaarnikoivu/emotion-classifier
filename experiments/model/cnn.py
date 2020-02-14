import torch
import torch.nn.functional as F
import torch.nn as nn


class BertCNN(nn.Module):
    def __init__(self, bert, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.bert = bert
        embedding_dim = 768
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]

        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, s1=1):
        super(ConvBlock).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=s1, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequential(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super().__init__()

        s1 = 2 if downsample == 'resnet' else 1
        pool_stride = 2 if downsample else 1

        # architecture
        self.conv_block = ConvBlock(in_channels, out_channels, s1=2)
        self.pool = None
        if downsample == 'kmax':
            self.pool = lambda x: x.topk(x.size(2) // 2)[0]
        elif downsample == 'vgg':
            self.pool = nn.MaxPool1d(
                kernel_size=3, stride=pool_stride, padding=1)
        self.shortcut = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        residual = self.conv_block(x)
        if self.pool:
            residual = self.pool(residual)
        residual += self.shortcut(x)
        return residual


class BertVDCNN(nn.Module):
    def __init__(self, bert, num_classes):
        super().__init__()

        self.k = 8
        self.bert = bert
        embedding_dim = 768

        self.conv = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, "vgg"),
            ResBlock(64, 128),
            ResBlock(128, 128, "vgg"),
            ResBlock(128, 256),
            ResBlock(256, 256, "vgg"),
            ResBlock(256, 512),
            ResBlock(512, 512)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * self.k, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]

        embedded = self.transpose(1, 2)
        h = self.conv(embedded)
        h = self.res_blocks(h)
        h = h.topk(self.k)[0].view(64, -1)
        h = self.fc(h)
        return h

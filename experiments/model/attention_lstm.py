import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class AttentionLSTM(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()

        self.bert = bert
        self.batch_size = opt.batch_size
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim

        embedding_dim = 768

        self.lstm = nn.LSTM(embedding_dim, hidden_size=opt.hidden_dim)
        self.label = nn.Linear(opt.hidden_dim, opt.output_dim)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, text, batch_size=None):
        with torch.no_grad():
            embedded = self.bert(text)[0]

        embedded = embedded.permute(1, 0, 2)

        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(embedded, (h_0, c_0))
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)
        return self.label(attn_output)

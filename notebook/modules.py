import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def to_2D(tensor, dim):
    return tensor.contiguous().view(-1, dim)


def to_3D(tensor, batch, dim):
    return tensor.contiguous().view(batch, -1, dim)


def expand(tensor, target):
    return tensor.expand_as(target)


def padded_attn(tensor, mask):
    tensor = torch.mul(tensor, mask)
    #tensor = torch.mul(tensor, mask.float())
    #tensor = torch.div(tensor, expand(tensor.sum(dim=1), tensor))  # for 0.11
    tensor = torch.div(tensor, expand(tensor.sum(dim=1).unsqueeze(1), tensor)) # for 0.12
    return tensor.unsqueeze(-1)


class InitPrev(nn.Module):
    def __init__(self, hidden_dim):
        super(InitPrev, self).__init__()
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(self.hidden_dim, 1)
        self.Wu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wv = nn.Linear(self.hidden_dim, self.hidden_dim)

    # self.Vr = nn.Parameter(torch.FloatTensor(self.hidden_dim, 1))

    def forward(self, qry_h, qm):
        batch_size = qry_h.size(0)
        qry_len = qry_h.size(1)

        qry_enc = to_2D(qry_h, self.hidden_dim)
        qry_enc = to_3D(self.Wu(qry_enc), batch_size, self.hidden_dim)

        Vr = Variable(torch.ones(qry_len, self.hidden_dim))

        Vr_enc = self.Wv(to_2D(Vr, self.hidden_dim)).unsqueeze(0)
        Vr_enc = expand(Vr_enc, qry_enc)

        s = F.tanh(qry_enc + Vr_enc)

        s = self.linear(to_2D(s, self.hidden_dim))
        alpha = F.softmax(s.view(batch_size, -1))
        alpha = padded_attn(alpha, qm)
        alpha = expand(alpha, qry_h)

        prev_h = torch.mul(to_2D(alpha, self.hidden_dim), to_2D(qry_h, self.hidden_dim))
        prev_h = to_3D(prev_h, batch_size, qry_len).sum(dim=2).squeeze(-1)
        return prev_h


class AttnLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttnLayer, self).__init__()
        self.hidden_dim = hidden_dim

        self.Wy = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wh = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wr = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, 1)
        self.Wt = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, doc_i, qry_h, prev_rt, qm):
        '''
        qry_h: B x Q x H
        doc_i: B X H
        prev_rt: B X H
        qm: B x Q
        '''

        batch_size = qry_h.size(0)
        qry_h_2D = to_2D(qry_h, self.hidden_dim)  # (B x Q) x H
        doc_i = to_2D(doc_i, self.hidden_dim)  # (B x 1) x H
        prev_rt = to_2D(prev_rt, self.hidden_dim)  # (B x 1) x H

        qry_enc = to_3D(self.Wy(qry_h_2D), batch_size, self.hidden_dim)  # B x Q x H

        doc_enc = self.Wh(doc_i)  # B x H
        prev_rt_enc = self.Wr(prev_rt)  # B x H

        ctx_enc = expand((doc_enc + prev_rt_enc).unsqueeze(1), qry_enc)  # B x Q x H

        Mt = F.tanh(qry_enc + ctx_enc)  # B x Q x H
        Mt = self.linear(to_2D(Mt, self.hidden_dim))

        alpha = F.softmax(Mt.view(batch_size, -1))  # B x Q
        alpha = padded_attn(alpha, qm)  # B x Q

        w_qry = torch.bmm(qry_h.transpose(1, 2), alpha)
        prev_rt_enc2 = self.Wt(prev_rt)
        rt = w_qry + F.tanh(prev_rt_enc2)

        return rt
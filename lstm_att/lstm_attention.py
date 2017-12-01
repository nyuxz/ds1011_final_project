from torchtext import data, datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import re
import random
import numpy as np
import argparse
import sys


# add parameters
parser = argparse.ArgumentParser(description='lstm_attention')
parser.add_argument('--hidden_dim', default=50, type=int, help='hidden dim (default: 200)')
parser.add_argument('--batch_size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--learning_rate', default=0.004, type=float, help='learning rate (default: 0.05)')
parser.add_argument('--embedding_dim', default=300, type=int, help='embedding dim (default: 300)')
parser.add_argument('--device', help='use GPU', default= None)
parser.add_argument('--lstm_att', help='save encoder', default= 'lstm_att.pt')
parser.add_argument('--pretrained_embed', default= 'glove.6B.300d', type = str, help='pretrained_embed')
parser.add_argument('--weight_decay', default= 0.0, type = float, help='weight_decay')
parser.add_argument('--num_layers', default=1, type=int, help='num_layers')
parser.add_argument('--bidirectional', default=True, type=bool, help='bidirectional')
parser.add_argument('--dropout', default= 0.0, type = float, help='dropout rate')




args = parser.parse_args()


use_cuda = torch.cuda.is_available()


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

        if use_cuda:
            Vr = Variable(torch.ones(qry_len, self.hidden_dim)).cuda()
        else:
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

class Embed(nn.Module):
    def __init__(self, W_emb, vocab_size, embed_dim, train_emb,dropout):
        super(Embed, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        if W_emb is not None:
            self.embed.weight = nn.Parameter(W_emb)
        if train_emb == False:
            self.embed.requires_grad = False

    def forward(self, doc, qry):
        doc = self.embed(doc)  # B x D x H
        doc = self.dropout(doc)
        qry = self.embed(qry)  # B x Q x H
        qry = self.dropout(qry)
        return doc, qry


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU can be changed to LSTM for experiments
        self.d_gru = nn.GRU(embed_size, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=args.bidirectional)
        self.q_gru = nn.GRU(embed_size, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=args.bidirectional)

        self.linear_d = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_q = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, doc, qry, qry_h0):
        batch_size = doc.size(0)

        qry_h, doc_h0 = self.q_gru(qry, qry_h0)  # B x Q x H
        doc_h, _ = self.d_gru(doc, doc_h0)  # B x D x H

        qry_h = self.linear_q(to_2D(qry_h, self.hidden_dim * 2))
        qry_h = to_3D(qry_h, batch_size, self.hidden_dim)

        doc_h = self.linear_d(to_2D(doc_h, self.hidden_dim * 2))
        doc_h = to_3D(doc_h, batch_size, self.hidden_dim)

        return doc_h, qry_h

    def init_hidden(self, batch_size):
        hidden = next(self.parameters()).data
        if(args.bidirectional == True):
            num_directions = 2
        else:
            num_directions = 1

        if use_cuda:
            tmp = Variable(hidden.new(self.num_layers * num_directions, batch_size, self.hidden_dim).zero_()).cuda()
        else:
            tmp = Variable(hidden.new(self.num_layers * num_directions, batch_size, self.hidden_dim).zero_())
        return tmp


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

        self.initprev = InitPrev(self.hidden_dim)
        self.attn = AttnLayer(self.hidden_dim)

    def forward(self, doc_h, qry_h, qm):
        batch_size = doc_h.size(0)
        doc_len = doc_h.size(1)
        prev_rt_init = self.initprev(qry_h, qm)

        for i in range(doc_len):
            if i == 0:
                prev_rt = prev_rt_init
            prev_rt = self.attn(doc_h[:, i, :], qry_h, prev_rt, qm)

        return prev_rt


class OutputLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(OutputLayer, self).__init__()
        self.hidden_dim = hidden_dim

        self.Wp = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wx = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, 3)

    def forward(self, rN, hN):
        rn_enc = to_2D(rN, self.hidden_dim)
        hn_enc = to_2D(hN, self.hidden_dim)

        rn_enc = self.Wp(rn_enc)
        hn_enc = self.Wx(hn_enc)

        h = F.tanh(rn_enc + hn_enc)
        h = self.linear(h)
        return h


class Entailment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, W_emb=None, dropout=0.3, num_layers=1, train_emb=True):
        super(Entailment, self).__init__()

        self.embed = Embed(W_emb, vocab_size, embed_dim, train_emb,dropout)
        self.encoder = Encoder(embed_dim, hidden_dim, num_layers)
        self.attention = Attention(hidden_dim)
        self.out = OutputLayer(hidden_dim)
        

    def forward(self, doc, qry, dm, qm):
        batch_size = doc.size(0)

        # Embedding the matrix
        d_emb, q_emb = self.embed(doc, qry)
        qry_h0 = self.encoder.init_hidden(batch_size)
        doc_h, qry_h = self.encoder(d_emb, q_emb, qry_h0)

        rt = self.attention(doc_h, qry_h, qm)
        output = self.out(rt, doc_h[:, -1, :])

        return output


def training_loop(model, loss, optimizer, train_iter, dev_iter, embed_dim, hidden_dim,lr):
    step = 0
    best_dev_acc = 0
    anneal_counter = 0
    while step <= num_train_steps:

        model.train()

        for batch in train_iter:
            premise = batch.premise.transpose(0, 1)
            hypothesis = batch.hypothesis.transpose(0, 1)
            labels = batch.label - 1
            model.zero_grad()

            if use_cuda:
                output = model(premise.cuda(), hypothesis.cuda(), embed_dim, hidden_dim)
            else:
                output = model(premise, hypothesis, embed_dim, hidden_dim)


            if use_cuda:
                lossy = loss(output, labels.cuda())
            else:
                lossy = loss(output, labels)


            lossy.backward()
            optimizer.step()


            if step % 100 == 0:
                dev_acc = evaluate(model, dev_iter, embed_dim, hidden_dim)
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    torch.save(model.state_dict(), args.lstm_att)
                    anneal_counter = 0
                if dev_acc <= best_dev_acc:
                        anneal_counter += 1
                        if anneal_counter == 100:
                            print('Annealing learning rate')
                            lr /= 2
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                            anneal_counter = 0
                print("Step %i; Loss %f; Dev acc %f; Best dev acc %f; learning rate %f" % (step, lossy.data[0], dev_acc, best_dev_acc,lr))
                sys.stdout.flush()
            if step >= num_train_steps:
                return best_dev_acc
            step += 1


def evaluate(model, data_iter, embed_dim, hidden_dim):
    model.eval()
    correct = 0
    total = 0
    for batch in data_iter:
        premise = batch.premise.transpose(0,1)
        hypothesis = batch.hypothesis.transpose(0,1)
        labels = (batch.label-1).data

        if use_cuda:
            output = model(premise.cuda(), hypothesis.cuda(), embed_dim, hidden_dim)
        else:
            output = model(premise, hypothesis, embed_dim, hidden_dim)

        if use_cuda:
            output.cpu()

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)

        if use_cuda:
            correct += (predicted == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    model.train()
    return correct / float(total)


def main():

    # get data
    inputs = datasets.snli.ParsedTextField(lower=True)
    answers = data.Field(sequential=False)
    train, dev, test = datasets.SNLI.splits(inputs, answers)
    inputs.build_vocab(train, vectors=args.pretrained_embed)
    answers.build_vocab(train)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size= args.batch_size, device=args.device)


    global num_train_steps
    vocab_size = len(inputs.vocab)
    input_size = vocab_size
    num_train_steps = 1000000000

    word_vecs = inputs.vocab.vectors
    model = Entailment(vocab_size=vocab_size, embed_dim= args.embedding_dim, hidden_dim = args.hidden_dim, W_emb=word_vecs, dropout=args.dropout,
                       num_layers=args.num_layers, train_emb=False)

    if use_cuda:
        model.cuda()

    # Loss and Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay = args.weight_decay)

    # Train the model
    training_loop(model, loss, optimizer, train_iter, dev_iter, args.embedding_dim, args.hidden_dim,args.learning_rate)

if __name__ == '__main__':
    main()

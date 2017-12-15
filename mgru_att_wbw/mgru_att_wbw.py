from torchtext import data, datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import re
import random
import numpy as np
from recurrent_BatchNorm import recurrent_BatchNorm
from utils import *
import argparse

# add parameters
parser = argparse.ArgumentParser(description='mgru_att')
parser.add_argument('--learning_rate', default=0.001,
                    type=float, help='learning rate (default: 0.05)')
parser.add_argument('--weight_decay', default=0,
                    type=float, help='weight_decay')
parser.add_argument('--batch_size', default=30, type=int,
                    help='batch size (default: 32)')
parser.add_argument('--embedding_dim', default=300, type=int,
                    help='embedding dim (default: 300)')
parser.add_argument('--hidden_dim', default=150, type=int,
                    help='hidden dim (default: 150)')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
parser.add_argument('--pretrained_embed',
                    default='glove.6B.300d', type=str, help='pretrained_embed')
parser.add_argument('--save_model', help='save encoder', default='mgru_att.pt')

args = parser.parse_args()


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
PAD_TOKEN = 0


if(use_cuda):
    device = None
else:
    device = -1


class RTE(nn.Module):
    def __init__(self, input_size, w_emb, EMBEDDING_DIM, HIDDEN_DIM):
        super(RTE, self).__init__()
        self.n_embed = EMBEDDING_DIM
        self.n_dim = HIDDEN_DIM if HIDDEN_DIM % 2 == 0 else HIDDEN_DIM - 1
        self.n_out = 3
        self.embedding = nn.Embedding(input_size, self.n_embed, padding_idx=1).type(dtype)
        self.embedding.weight = nn.Parameter(w_emb)
        self.embedding.requires_grad = False
        self.p_gru = nn.GRU(self.n_embed, self.n_dim,
                            bidirectional=False).type(dtype)
        self.h_gru = nn.GRU(self.n_embed, self.n_dim,
                            bidirectional=False).type(dtype)
        self.out = nn.Linear(self.n_dim, self.n_out).type(dtype)

        # Attention Parameters
        self.W_y = nn.Parameter(torch.randn(self.n_dim, self.n_dim).cuda(
        )) if use_cuda else nn.Parameter(torch.randn(self.n_dim, self.n_dim))  # n_dim x n_dim
        self.register_parameter('W_y', self.W_y)
        self.W_h = nn.Parameter(torch.randn(self.n_dim, self.n_dim).cuda(
        )) if use_cuda else nn.Parameter(torch.randn(self.n_dim, self.n_dim))  # n_dim x n_dim
        self.register_parameter('W_h', self.W_h)
        self.W_r = nn.Parameter(torch.randn(self.n_dim, self.n_dim).cuda(
        )) if use_cuda else nn.Parameter(torch.randn(self.n_dim, self.n_dim))  # n_dim x n_dim
        self.register_parameter('W_r', self.W_r)
        self.W_alpha = nn.Parameter(torch.randn(self.n_dim, 1).cuda(
        )) if use_cuda else nn.Parameter(torch.randn(self.n_dim, 1))  # n_dim x 1
        self.register_parameter('W_alpha', self.W_alpha)

        '''

        if WBW_ATTN:
            # Since the word by word attention layer is a simple rnn, it suffers from the gradient exploding problem
            # A way to circumvent that is having orthonormal initialization of the weight matrix
            _W_t = np.random.randn(self.n_dim, self.n_dim)
            _W_t_ortho, _ = np.linalg.qr(_W_t)
            self.W_t = nn.Parameter(torch.Tensor(_W_t_ortho).cuda()) if use_cuda else nn.Parameter(torch.Tensor(_W_t_ortho))  # n_dim x n_dim
            self.register_parameter('W_t', self.W_t)
            self.batch_norm_h_r = recurrent_BatchNorm(self.n_dim, 30).type(dtype) # 'MAX_LEN' = 30
            self.batch_norm_r_r = recurrent_BatchNorm(self.n_dim, 30).type(dtype)
        '''

        # Match GRU parameters.
        self.m_gru = nn.GRU(self.n_dim + self.n_dim,
                            self.n_dim, bidirectional=False).type(dtype)

    def init_hidden(self, batch_size):
        hidden_p = Variable(torch.zeros(1, batch_size, self.n_dim).type(dtype))
        hidden_h = Variable(torch.zeros(1, batch_size, self.n_dim).type(dtype))
        return hidden_p, hidden_h

    def attn_gru_init_hidden(self, batch_size):
        r_0 = Variable(torch.zeros(batch_size, self.n_dim).type(dtype))
        return r_0

    def mask_mult(self, o_t, o_tm1, mask_t):
        '''
            o_t : batch x n
            o_tm1 : batch x n
            mask_t : batch x 1
        '''
        # return (mask_t.expand(*o_t.size()) * o_t) + ((1. - mask_t.expand(*o_t.size())) * (o_tm1))
        return (o_t * mask_t) + (o_tm1 * (1. - mask_t))

    def _gru_forward(self, gru, encoded_s, mask_s, h_0):
        '''
        inputs :
            gru : The GRU unit for which the forward pass is to be computed
            encoded_s : T x batch x n_embed
            mask_s : T x batch
            h_0 : 1 x batch x n_dim
        outputs :
            o_s : T x batch x n_dim
            h_n : 1 x batch x n_dim
        '''
        seq_len = encoded_s.size(0)
        batch_size = encoded_s.size(1)
        o_s = Variable(torch.zeros(
            seq_len, batch_size, self.n_dim).type(dtype))
        h_tm1 = h_0.squeeze(0)  # batch x n_dim
        o_tm1 = None

        for ix, (x_t, mask_t) in enumerate(zip(encoded_s, mask_s)):
            '''
                x_t : batch x n_embed
                mask_t : batch,
            '''
            o_t, h_t = gru(x_t.unsqueeze(0), h_tm1.unsqueeze(0)
                           )  # o_t : 1 x batch x n_dim
            # h_t : 1 x batch x n_dim
            mask_t = mask_t.unsqueeze(1)  # batch x 1
            h_t = self.mask_mult(h_t[0], h_tm1, mask_t)

            if o_tm1 is not None:
                o_t = self.mask_mult(o_t[0], o_tm1, mask_t)
            o_tm1 = o_t[0] if o_tm1 is None else o_t
            h_tm1 = h_t
            o_s[ix] = o_t

        return o_s, h_t.unsqueeze(0)

    def _attention_forward(self, Y, mask_Y, h, r_tm1=None):
        '''
        Computes the Attention Weights over Y using h (and r_tm1 if given)
        Returns an attention weighted representation of Y, and the alphas
        inputs:
            Y : T x batch x n_dim
            mask_Y : T x batch
            h : batch x n_dim
            r_tm1 : batch x n_dim
        params:
            W_y : n_dim x n_dim
            W_h : n_dim x n_dim
            W_r : n_dim x n_dim
            W_alpha : n_dim x 1
        outputs :
            r = batch x n_dim
            alpha : batch x T
        '''
        Y = Y.transpose(1, 0)  # batch x T x n_dim
        mask_Y = mask_Y.transpose(1, 0)  # batch x T

        Wy = torch.bmm(Y, self.W_y.unsqueeze(0).expand(
            Y.size(0), *self.W_y.size()))  # batch x T x n_dim
        Wh = torch.mm(h, self.W_h)  # batch x n_dim
        if r_tm1 is not None:
            W_r_tm1 = torch.mm(r_tm1, self.W_r)
            Wh += W_r_tm1
        M = torch.tanh(Wy + Wh.unsqueeze(1).expand(Wh.size(0),
                                                   Y.size(1), Wh.size(1)))  # batch x T x n_dim
        alpha = torch.bmm(M, self.W_alpha.unsqueeze(0).expand(
            Y.size(0), *self.W_alpha.size())).squeeze(-1)  # batch x T
        # To ensure probability mass doesn't fall on non tokens
        alpha = alpha + (-1000.0 * (1. - mask_Y))
        alpha = F.softmax(alpha)
        return torch.bmm(alpha.unsqueeze(1), Y).squeeze(1), alpha

    def _attn_gru_forward(self, o_h, mask_h, r_0, o_p, mask_p):
        '''
        inputs:
            o_h : T x batch x n_dim : The hypothesis
            mask_h : T x batch
            r_0 : batch x n_dim :
            o_p : T x batch x n_dim : The premise. Will attend on it at every step
            mask_p : T x batch : the mask for the premise
        params:
            m_gru params
        outputs:
            r : batch x n_dim : the last state of the rnn
            alpha_vec : T x batch x T the attn vec at every step
        '''
        seq_len_h = o_h.size(0)
        batch_size = o_h.size(1)
        seq_len_p = o_p.size(0)
        alpha_vec = Variable(torch.zeros(
            seq_len_h, batch_size, seq_len_p).type(dtype))
        r_tm1 = r_0
        for ix, (h_t, mask_t) in enumerate(zip(o_h, mask_h)):
            '''
                h_t : batch x n_dim
                mask_t : batch,
            '''
            a_t, alpha = self._attention_forward(
                o_p, mask_p, h_t, r_tm1)   # a_t : batch x n_dim
            # alpha : batch x T
            alpha_vec[ix] = alpha
            m_t = torch.cat([a_t, h_t], dim=-1)
            r_t, _ = self.m_gru(m_t.unsqueeze(0), r_tm1.unsqueeze(0))

            mask_t = mask_t.unsqueeze(1)  # batch x 1
            r_t = self.mask_mult(r_t[0], r_tm1, mask_t)
            r_tm1 = r_t

        return r_t, alpha_vec

    def forward(self, premise, hypothesis, training=False):
        '''
        inputs:
            premise : batch x T
            hypothesis : batch x T
        outputs :
            pred : batch x num_classes
        '''
        self.train(training)
        batch_size = premise.size(0)

        mask_p = torch.ne(premise, 0).type(dtype)
        mask_h = torch.ne(hypothesis, 0).type(dtype)

        encoded_p = self.embedding(premise)  # batch x T x n_embed
        encoded_p = F.dropout(encoded_p, p=0.1, training=training)

        encoded_h = self.embedding(hypothesis)  # batch x T x n_embed
        encoded_h = F.dropout(encoded_h, p=0.1, training=training)

        encoded_p = encoded_p.transpose(1, 0)  # T x batch x n_embed
        encoded_h = encoded_h.transpose(1, 0)  # T x batch x n_embed

        mask_p = mask_p.transpose(1, 0)  # T x batch
        mask_h = mask_h.transpose(1, 0)  # T x batch

        h_p_0, h_n_0 = self.init_hidden(batch_size)  # 1 x batch x n_dim
        o_p, h_n = self._gru_forward(
            self.p_gru, encoded_p, mask_p, h_p_0)  # o_p : T x batch x n_dim
        # h_n : 1 x batch x n_dim

        o_h, h_n = self._gru_forward(
            self.h_gru, encoded_h, mask_h, h_n_0)  # o_h : T x batch x n_dim
        # h_n : 1 x batch x n_dim

        r_0 = self.attn_gru_init_hidden(batch_size)
        h_star, alpha_vec = self._attn_gru_forward(
            o_h, mask_h, r_0, o_p, mask_p)

        h_star = self.out(h_star)  # batch x num_classes

        '''
        if self.options['LAST_NON_LINEAR']:
            h_star = F.leaky_relu(h_star)  # Non linear projection
        '''
        pred = F.log_softmax(h_star)
        return pred

    def _get_numpy_array_from_variable(self, variable):
        '''
        Converts a torch autograd variable to its corresponding numpy array
        '''
        if use_cuda:
            return variable.cpu().data.numpy()
        else:
            return variable.data.numpy()


def training_loop(model, loss, optimizer, train_iter, dev_iter, lr):
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
                output = model(premise.cuda(), hypothesis.cuda())
            else:
                output = model(premise, hypothesis)
            if use_cuda:
                lossy = loss(output, labels.cuda())
            else:
                lossy = loss(output, labels)

            lossy.backward()
            optimizer.step()

            if step % 100 == 0:
                dev_acc = evaluate(model, dev_iter)
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    torch.save(model.state_dict(), args.save_model)
                    anneal_counter = 0
                if dev_acc <= best_dev_acc:
                    anneal_counter += 1
                    if anneal_counter == 100:
                        print('Annealing learning rate')
                        lr = lr * 0.95  # learning rate decay ratio (not sure)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        anneal_counter = 0
                print("Step %i; Loss %f; Dev acc %f; Best dev acc %f; learning rate %f" % (step, lossy.data[0], dev_acc, best_dev_acc, lr))
                sys.stdout.flush()

            if step >= num_train_steps:
                return best_dev_acc
            step += 1


def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0
    for batch in data_iter:
        premise = batch.premise.transpose(0, 1)
        hypothesis = batch.hypothesis.transpose(0, 1)
        labels = (batch.label - 1).data

        if use_cuda:
            output = model(premise.cuda(), hypothesis.cuda())
        else:
            output = model(premise, hypothesis)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)

        if use_cuda:
            correct += (predicted == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    model.train()
    return correct / float(total)


def main():
    
    inputs = datasets.snli.ParsedTextField(lower=True)
    answers = data.Field(sequential=False)
    train, dev, test = datasets.SNLI.splits(inputs, answers)
    
    # get input embeddings
    inputs.build_vocab(train, vectors=args.pretrained_embed)
    answers.build_vocab(train)

    # global params
    global input_size, num_train_steps
    vocab_size = len(inputs.vocab)
    input_size = vocab_size
    num_train_steps = 50000000
    word_vecs = inputs.vocab.vectors


    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=device)

    model = RTE(input_size, w_emb = word_vecs, EMBEDDING_DIM=args.embedding_dim,
                HIDDEN_DIM=args.hidden_dim)
    # Loss
    loss = nn.NLLLoss()

    # Optimizer
    para2 = model.parameters()
    optimizer = torch.optim.Adam(para2, lr=args.learning_rate, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    # Train the model
    training_loop(model, loss, optimizer, train_iter,
                  dev_iter, args.learning_rate)


if __name__ == '__main__':
    main()

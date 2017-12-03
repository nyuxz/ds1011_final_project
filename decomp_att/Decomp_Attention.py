from torchtext import data, datasets
from torch.autograd import Variable
from data_loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import random
import argparse
import numpy as np
import sys


# add parameters
parser = argparse.ArgumentParser(description='decomposable_attention')
parser.add_argument('--batch_size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--encoder_path', default='encoder.pt', help='save encoder')
parser.add_argument('--model_path', default='model.pt', help='save model')
args = parser.parse_args()


class EmbedEncoder(nn.Module):

    def __init__(self, input_size, embedding_dim, hidden_dim, para_init):
        super(EmbedEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=100)
        self.input_linear = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.para_init = para_init

        '''initialize parameters'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)

    def forward(self, prem, hypo):
        batch_size = prem.size(0)

        prem_emb = self.embed(prem)
        hypo_emb = self.embed(hypo)

        prem_emb = prem_emb.view(-1, self.embedding_dim)
        hypo_emb = hypo_emb.view(-1, self.embedding_dim)

        prem_emb = self.input_linear(prem_emb).view(batch_size, -1, self.hidden_dim)
        hypo_emb = self.input_linear(hypo_emb).view(batch_size, -1, self.hidden_dim)

        return prem_emb, hypo_emb


# Decomposable Attention
class DecomposableAttention(nn.Module):
    # inheriting from nn.Module!

    def __init__(self, hidden_dim, num_labels, para_init):
        super(DecomposableAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = nn.Dropout(p=0.2)
        self.para_init = para_init

        # layer F, G, and H are feed forward nn with ReLu
        self.mlp_F = self.mlp(hidden_dim, hidden_dim)
        self.mlp_G = self.mlp(2 * hidden_dim, hidden_dim)
        self.mlp_H = self.mlp(2 * hidden_dim, hidden_dim)

        # final layer will not use dropout, so defining independently
        self.linear_final = nn.Linear(hidden_dim, num_labels, bias=True)

        '''initialize parameters'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                m.bias.data.normal_(0, self.para_init)

    def mlp(self, input_dim, output_dim):
        '''
        Define a feed forward neural network with ReLu activations
        '''
        feed_forward = []
        feed_forward.append(self.dropout)
        feed_forward.append(nn.Linear(input_dim, output_dim, bias=True))
        feed_forward.append(nn.ReLU())
        feed_forward.append(self.dropout)
        feed_forward.append(nn.Linear(output_dim, output_dim, bias=True))
        feed_forward.append(nn.ReLU())
        return nn.Sequential(*feed_forward)

    def forward(self, prem_emb, hypo_emb):

        '''Input layer'''
        len_prem = prem_emb.size(1)
        len_hypo = hypo_emb.size(1)

        '''Attend'''
        f_prem = self.mlp_F(prem_emb.view(-1, self.hidden_dim))
        f_hypo = self.mlp_F(hypo_emb.view(-1, self.hidden_dim))

        f_prem = f_prem.view(-1, len_prem, self.hidden_dim)
        f_hypo = f_hypo.view(-1, len_hypo, self.hidden_dim)

        e_ij = torch.bmm(f_prem, torch.transpose(f_hypo, 1, 2))
        beta_ij = F.softmax(e_ij.view(-1, len_hypo)).view(-1, len_prem, len_hypo)
        beta_i = torch.bmm(beta_ij, hypo_emb)

        e_ji = torch.transpose(e_ij.contiguous(), 1, 2)
        e_ji = e_ji.contiguous()
        alpha_ji = F.softmax(e_ji.view(-1, len_prem)).view(-1, len_hypo, len_prem)
        alpha_j = torch.bmm(alpha_ji, prem_emb)

        '''Compare'''
        concat_1 = torch.cat((prem_emb, beta_i), 2)
        concat_2 = torch.cat((hypo_emb, alpha_j), 2)
        compare_1 = self.mlp_G(concat_1.view(-1, 2 * self.hidden_dim))
        compare_2 = self.mlp_G(concat_2.view(-1, 2 * self.hidden_dim))
        compare_1 = compare_1.view(-1, len_prem, self.hidden_dim)
        compare_2 = compare_2.view(-1, len_hypo, self.hidden_dim)

        '''Aggregate'''
        v_1 = torch.sum(compare_1, 1)
        v_1 = torch.squeeze(v_1, 1)
        v_2 = torch.sum(compare_2, 1)
        v_2 = torch.squeeze(v_2, 1)
        v_concat = torch.cat((v_1, v_2), 1)
        y_pred = self.mlp_H(v_concat)

        '''Final layer'''
        out = F.log_softmax(self.linear_final(y_pred))

        return out


def train(batch_size, use_shrinkage, num_train_steps, encoder_path, model_path, to_lower):
    vocab, word_embeddings, word_to_index, index_to_word = load_embedding_and_build_vocab('../data/glove.6B.300d.txt')

    training_set = process_snli('../data/snli_1.0_train.jsonl', word_to_index, to_lower)
    train_iter = batch_iter(dataset=training_set, batch_size=batch_size, shuffle=True)
    dev_set = process_snli('../data/snli_1.0_dev.jsonl', word_to_index, to_lower)
    dev_iter = batch_iter(dataset=dev_set, batch_size=batch_size, shuffle=True)

    num_batch = len(dev_set) // batch_size

    use_cuda = torch.cuda.is_available()

    # Normalize embedding vector (l2-norm = 1)
    word_embeddings[100, :] = np.ones(300)
    word_embeddings = (word_embeddings.T / np.linalg.norm(word_embeddings, ord=2, axis=1)).T
    word_embeddings[100, :] = np.zeros(300)

    # Encoder and Model
    input_encoder = EmbedEncoder(input_size=word_embeddings.shape[0], embedding_dim=300, hidden_dim=200, para_init=0.01)
    input_encoder.embed.weight.data.copy_(torch.from_numpy(word_embeddings))
    input_encoder.embed.weight.requires_grad = False
    model = DecomposableAttention(hidden_dim=200, num_labels=3, para_init=0.01)

    if use_cuda:
        input_encoder.cuda()
        model.cuda()

    # Optimizer
    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters())
    para2 = model.parameters()
    input_optimizer = torch.optim.Adagrad(para1, lr=0.05, weight_decay=0)
    optimizer = torch.optim.Adagrad(para2, lr=0.05, weight_decay=0)

    # Initialize the optimizer
    for group in input_optimizer.param_groups:
        for p in group['params']:
            state = input_optimizer.state[p]
            state['sum'] += 0.1
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            state['sum'] += 0.1

    # Loss
    loss = nn.NLLLoss()

    best_dev_acc = 0

    for step, (label, premise, hypothesis) in enumerate(train_iter):

        input_encoder.train()
        model.train()

        if use_cuda:
            premise_var = Variable(torch.LongTensor(premise).cuda())
            hypothesis_var = Variable(torch.LongTensor(hypothesis).cuda())
            label_var = Variable(torch.LongTensor(label).cuda())
        else:
            premise_var = Variable(torch.LongTensor(premise))
            hypothesis_var = Variable(torch.LongTensor(hypothesis))
            label_var = Variable(torch.LongTensor(label))
        
        input_encoder.zero_grad()
        model.zero_grad()

        prem_emb, hypo_emb = input_encoder(premise_var, hypothesis_var)
        output = model(prem_emb, hypo_emb)

        lossy = loss(output, label_var)
        lossy.backward()

        # Shrinkage
        if use_shrinkage is True:
            grad_norm = 0.
            for m in input_encoder.modules():
                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    if m.bias is not None:
                        grad_norm += m.bias.grad.data.norm() ** 2
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    if m.bias is not None:
                        grad_norm += m.bias.grad.data.norm() ** 2
            grad_norm ** 0.5
            shrinkage = 5 / (grad_norm + 1e-6)
            if shrinkage < 1:
                for m in input_encoder.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                        m.bias.grad.data = m.bias.grad.data * shrinkage

        input_optimizer.step()
        optimizer.step()
        
        if step % 100 == 0:
            dev_acc = evaluate(model, input_encoder, dev_iter, num_batch, use_cuda)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(input_encoder.state_dict(), encoder_path)
                torch.save(model.state_dict(), model_path)
            print('Step %i; Loss %f; Dev acc %f; Best dev acc %f;' % (step, lossy.data[0], dev_acc, best_dev_acc))
            sys.stdout.flush()
        if step >= num_train_steps:
            print('Step %i; Loss %f; Dev acc %f; Best dev acc %f;' % (step, lossy.data[0], dev_acc, best_dev_acc))


def evaluate(model, input_encoder, data_iter, num_batch, use_cuda):
    input_encoder.eval()
    model.eval()
    correct = 0
    total = 0

    for _ in range(num_batch):
        label, premise, hypothesis = next(data_iter)

        if use_cuda:
            premise_var = Variable(torch.LongTensor(premise).cuda())
            hypothesis_var = Variable(torch.LongTensor(hypothesis).cuda())
            label_var = Variable(torch.LongTensor(label).cuda())
        else:
            premise_var = Variable(torch.LongTensor(premise))
            hypothesis_var = Variable(torch.LongTensor(hypothesis))
            label_var = Variable(torch.LongTensor(label))

        prem_emb, hypo_emb = input_encoder(premise_var, hypothesis_var)
        output = model(prem_emb, hypo_emb)

        if use_cuda:
            output.cpu()

        _, predicted = torch.max(output.data, 1)
        total += len(label)
        correct += (predicted == label_var.data).sum()

    input_encoder.train()
    model.train()
    return correct / float(total)


if __name__ == '__main__':
    train(batch_size=args.batch_size, use_shrinkage=False, num_train_steps=50000000, encoder_path=args.encoder_path, model_path=args.model_path, to_lower=True)

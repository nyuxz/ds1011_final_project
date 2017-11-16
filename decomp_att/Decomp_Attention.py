from torchtext import data, datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import re
import random
import argparse
import numpy as np
import sys


# add parameters 
parser = argparse.ArgumentParser(description='decomposable_attention')
parser.add_argument('--num_labels', default=3, type=int, help='number of labels (default: 3)')
parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dim (default: 200)')
# parser.add_argument('--batch_size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--learning_rate', default=0.05, type=int, help='learning rate (default: 0.05)')
parser.add_argument('--embedding_dim', default=300, type=int, help='embedding dim (default: 300)')
parser.add_argument('--para_init', help='parameter initialization gaussian', type=float, default=0.01)


class EmbedEncoder(nn.Module):
    
    def __init__(self, input_size, embedding_dim, hidden_dim, para_init):
        super(EmbedEncoder, self).__init__()     

        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=1)
        self.input_linear = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.para_init = para_init   

        '''initialize parameters'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)

    def forward(self, prem, hypo):
        prem_emb = self.embed(prem)
        hypo_emb = self.embed(hypo)
        prem_emb = self.input_linear(prem_emb)
        hypo_emb = self.input_linear(hypo_emb)
        return prem_emb, hypo_emb


# A Multi-Layer Perceptron (MLP)
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
        self.linear_final = nn.Linear(hidden_dim, num_labels, bias=False)

        '''initialize parameters'''
        for m in self.modules():
            # print m
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                # m.bias.data.normal_(0, self.para_init)

    def mlp(self, input_dim, output_dim):
        '''
        function define a feed forward neural network with ReLu activations 
        @input: dimension specifications
        
        ToDo: 
            1. bias 
            2. args of dropout(maybe) 
            3. initialize para   
        '''
        feed_forward = []
        feed_forward.append(self.dropout)
        feed_forward.append(nn.Linear(input_dim, output_dim, bias=False))
        feed_forward.append(nn.ReLU())
        feed_forward.append(self.dropout)
        feed_forward.append(nn.Linear(output_dim, output_dim, bias=False))
        feed_forward.append(nn.ReLU()) 
        return nn.Sequential(*feed_forward)
   
    def forward(self, prem_emb, hypo_emb):

        '''Input layer'''
        
        '''Attend'''
        f_prem = self.mlp_F(prem_emb)
        f_hypo = self.mlp_F(hypo_emb)
        e_ij = torch.bmm(f_prem, torch.transpose(f_hypo, 1, 2))
        beta_ij = F.softmax(e_ij)
        beta_i = torch.bmm(beta_ij, hypo_emb)
        e_ji = torch.transpose(e_ij, 1, 2)
        alpha_ji = F.softmax(e_ji)
        alpha_j = torch.bmm(alpha_ji, prem_emb)
                  
        '''Compare'''
        concat_1 = torch.cat((prem_emb, beta_i), 2)       
        concat_2 = torch.cat((hypo_emb, alpha_j), 2)       
        compare_1 = self.mlp_G(concat_1)
        compare_2 = self.mlp_G(concat_2)
              
        '''Aggregate'''
        v_1 = torch.sum(compare_1, 1)
        v_2 = torch.sum(compare_2, 1)
        v_concat = torch.cat((v_1, v_2), 1)    
        y_pred = self.mlp_H(v_concat)
        
        '''Final layer'''
        out = F.log_softmax(self.linear_final(y_pred))

        return out


def training_loop(model, input_encoder, loss, optimizer, input_optimizer, train_iter, dev_iter):
    step = 0
    best_dev_acc = 0
    for i in range(num_train_steps): 
        input_encoder.train()         
        model.train()    
        for batch in train_iter:
            premise = batch.premise.transpose(0, 1)
            hypothesis = batch.hypothesis.transpose(0, 1)
            labels = batch.label - 1           
            input_encoder.zero_grad()
            model.zero_grad()
            prem_emb, hypo_emb = input_encoder(premise.cuda(), hypothesis.cuda())
            output = model(prem_emb, hypo_emb)
            lossy = loss(output, labels)
            lossy.backward()    
            #To-Do:grad 
            input_optimizer.step()
            optimizer.step()
            if step % 100 == 0:
                dev_acc = evaluate(model, input_encoder, dev_iter)
                print("Step %i; Loss %f; Dev acc %f" % (step, lossy.data[0], dev_acc))
                sys.stdout.flush()
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    torch.save(input_encoder.state_dict, 'input_encoder.pt')
                    torch.save(model.state_dict(), 'decomp_atten.pt')
            step += 1
    return best_dev_acc


def evaluate(model, input_encoder, data_iter):
    input_encoder.eval()
    model.eval()
    correct = 0
    total = 0
    for batch in data_iter: 
        premise = batch.premise.transpose(0, 1)
        hypothesis = batch.hypothesis.transpose(0, 1)
        labels = (batch.label - 1).data
        prem_emb, hypo_emb = input_encoder(premise.cuda(), hypothesis.cuda())
        output = model(prem_emb, hypo_emb)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    input_encoder.train()
    model.train()
    return correct / float(total)


def main():

    # get data
    inputs = datasets.snli.ParsedTextField(lower=True)
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers)

    # get input embeddings
    inputs.build_vocab(train, vectors='glove.6B.300d')
    answers.build_vocab(train)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=4, device=None)

    # global params 
    global input_size, num_train_steps
    vocab_size = len(inputs.vocab)
    input_size = vocab_size
    num_train_steps = 1000000
    args = parser.parse_args()

    #define model

    word_vecs = inputs.vocab.vectors 
    #word_vecs = torch.from_numpy(word_vecs)
    input_encoder = EmbedEncoder(input_size, args.embedding_dim, args.hidden_dim, args.para_init)
    input_encoder.embed.weight.data.copy_(word_vecs)
    input_encoder.embed.weight.requires_grad = False
    input_encoder.cuda()

    model = DecomposableAttention(args.hidden_dim, args.num_labels, args.para_init)
    model.cuda()

    #Loss
    loss = nn.CrossEntropyLoss()

    # Optimizer

    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters()) 
    para2 = model.parameters()

    input_optimizer = torch.optim.Adagrad(para1, lr=args.learning_rate)
    optimizer = torch.optim.Adagrad(para2, lr=args.learning_rate)
    
    #Train the model
    best_dev_acc = training_loop(model, input_encoder, loss, optimizer, input_optimizer, train_iter, dev_iter)
    print(best_dev_acc)


if __name__ == '__main__':
    main()




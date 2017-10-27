from torchtext import data, datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import re
import random
import argparse


parser = argparse.ArgumentParser(description='cbow+mlp')
parser.add_argument('--num_labels', default=3, type=int, help='number of labels (default: 3)')
parser.add_argument('--hidden_dim', default=10, type=int, help='number of hidden dim (default: 10)')

inputs = datasets.snli.ParsedTextField(lower=True)
labels = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, labels)

inputs.build_vocab(train)
labels.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=64, device=-1)


# Continuous Bag of Words (CBOW) + Multi-Layer Perceptron (MLP)
class CBOW_MLP(nn.Module): 

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        """
        @param vocab_size: size of the vocabulary
        @param embedding_dim: size of the word embedding
        @param hidden_dim: size of the hidden layer
        @param num_labels: number of labels
        """
        super(CBOW_MLP, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(2 * embedding_dim, hidden_dim) 
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, num_labels)
        self.init_weights()

    def forward(self, prem, hypo):
        """
        @param prem: a long tensor of size (batch_size * sentence_length)
        @param hypo: a long tensor of size (batch_size * sentence_length)
        """
        emb_prem = self.embed(prem).mean(1)
        emb_hypo = self.embed(hypo).mean(1)
        emb_concat = torch.cat([emb_prem, emb_hypo], 1)
        out = self.dropout(emb_concat)
        out = F.relu(self.linear_1(out))
        out = F.relu(self.linear_2(out))
        out = self.dropout(self.linear_3(out))
        return F.log_softmax(out)

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_1, self.linear_2, self.linear_3]
        em_layer = [self.embed]
        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


def training_loop(model, loss, optimizer, train_iter, dev_iter, max_num_train_steps):
    step = 0
    for i in range(max_num_train_steps):
        model.train()
        for batch in train_iter:
            premise = batch.premise.transpose(0, 1)
            hypothesis = batch.hypothesis.transpose(0, 1)
            labels = batch.label - 1
            model.zero_grad()
            output = model(premise, hypothesis)
            lossy = loss(output, labels)
            lossy.backward()
            optimizer.step()
            if step % 10 == 0:
                print( "Step %i; Loss %f; Dev acc %f" % (step, lossy.data[0], evaluate(model, dev_iter)))
            step += 1


def evaluate(model, data_iter):
    """
    @param model: 
    @param data_iter: data loader for the dataset to test against
    """
    model.eval()
    correct = 0
    total = 0
    for batch in data_iter:
        premise = batch.premise.transpose(0, 1)
        hypothesis = batch.hypothesis.transpose(0, 1)
        labels = (batch.label - 1).data
        output = model(premise, hypothesis)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    model.train()
    return correct / float(total)


def main():
    global args, max_num_train_steps, learning_rate, batch_size, embedding_dim
    args = parser.parse_args()

    vocab_size = len(inputs.vocab)
    #num_labels = 3
    #hidden_dim = 50
    embedding_dim = 300
    batch_size = 32
    learning_rate = 0.004
    max_num_train_steps = 1000
    
    model = CBOW_MLP(vocab_size, embedding_dim, args.hidden_dim, args.num_labels)   
    # Loss and Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    training_loop(model, loss, optimizer, train_iter, dev_iter, max_num_train_steps)


if __name__ == '__main__':
    main()

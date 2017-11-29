from lstm_attention import *
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

# How to run test_model.py
# run in terminal
# module load pytorch/python3.5/0.2.0_3
# python3 test_model.py --model_name 'model_200_32_0004_lstm.pt' --batch_size 32 --hidden_dim 200 --embedding_dim 300


# declare hyperparameters
parser = argparse.ArgumentParser(description='test model')
parser.add_argument('--model_name', type = str, help='saved model name')
parser.add_argument('--batch_size', type = int, help='batch size')
parser.add_argument('--hidden_dim', type = int, help='hidden dim (default: 200)')
parser.add_argument('--embedding_dim', type = int, help='embedding dim (default: 300)')

args_2 = parser.parse_args()


# load data 
inputs = datasets.snli.ParsedTextField(lower=True)
answers = data.Field(sequential=False)
train, dev, test = datasets.SNLI.splits(inputs, answers)
inputs.build_vocab(train, vectors='glove.6B.300d')
answers.build_vocab(train)
vocab_size = len(inputs.vocab)
word_vecs = inputs.vocab.vectors
train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size= args_2.batch_size, device= -1)

# load model

model = Entailment(vocab_size=vocab_size, embed_dim= args_2.embedding_dim, hidden_dim = args_2.hidden_dim, 
    W_emb=word_vecs, p=0.3, num_layers=1, train_emb=False)
model.load_state_dict(torch.load(args_2.model_name, map_location = lambda storage, loc: storage))
model.cpu()

# test model 
test_acc = evaluate(model, test_iter, embed_dim= args_2.embedding_dim, hidden_dim = args_2.hidden_dim)
print('Accuracy of the network on the test data: %f' % (100 * test_acc))

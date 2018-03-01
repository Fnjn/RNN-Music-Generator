#!/usr/bin/env python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_size, drop_ratio):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input2hidden = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, dropout=drop_ratio)
        self.hidden2tag = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h = torch.zeros(self.num_layers,1,self.hidden_dim)
        c = torch.zeros(self.num_layers,1,self.hidden_dim)
        return (Variable(h.cuda()), Variable(c.cuda()))

    def forward(self, x):
        x_len = len(x)
        x = self.input2hidden(x)
        x, self.hidden = self.lstm(x.view(x_len, 1, -1), self.hidden)
        x = self.hidden2tag(x.view(x_len, -1))
        return x


class PrepareData():

    def __init__(self, filename, batch_size, val_ratio=0.):
        self.filename = filename
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        self.load_dataset(filename, batch_size, val_ratio)

    def load_dataset(self, filename, batch_size, val_ratio=0.):
        with open(filename, 'r') as f:
            data = f.read()
        self.data = [data[i*batch_size:(i+1)*batch_size] for i in range(len(data)//batch_size)]

        n_data = len(self.data)
        n_val = int(n_data * self.val_ratio)
        self.train_data = self.data[n_val:]
        self.val_data = self.data[:n_val]

    def shuffle_data(self):
        n_data = len(self.data)
        n_val = int(n_data * self.val_ratio)
        perm = np.random.permutation(n_data)

        self.train_data = [self.data[i] for i in perm[n_val:]]
        self.val_data = [self.data[i] for i in perm[:n_val]]

    def dataset(self):
        return self.train_data, self.val_data

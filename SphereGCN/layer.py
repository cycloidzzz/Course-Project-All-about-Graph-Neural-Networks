import torch
import random
import numpy as np
from torch import nn
from torch.nn import functional as F
from utils import sparse_dropout


#advanced gcn conv
class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim,
               dropout,
               residue = True,
               active = F.relu,
               normed = False,
               bias = False
               ):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.output_dim = output_dim

        if input_dim != output_dim:
            residue = False

        self.residue = residue
        self.active = active

        self.fc1 = nn.Linear(input_dim, output_dim, bias = False)
        nn.init.kaiming_normal(self.fc1.weight)

        self.scale = nn.Parameter(torch.tensor([1.5000], dtype = torch.float32))

        self.normed = normed
        self.bias = None

        self.batchnorm_h = nn.BatchNorm1d(output_dim)

        if bias:
            self.bias = nn.Parameters(torch.randn(output_dim))

    def forward(self, inputs):

        x, adj1, adj2 = inputs

        y = F.dropout(x, self.dropout, training = self.training)

        y = self.fc1(y)
        y = torch.sparse.mm(adj1, y)

        if self.normed == 'sp_norm':
            N = y.shape[0]

            y_t = y - (y.sum(0)/(N)).repeat((N, 1)).view(y.shape)

            y_norm = y_t.pow(2).sum(1).view((N, 1)).mean(0).pow(-(1/2))

            y_norm = y_norm.repeat((N, 1))

            y = y_t * y_norm * (1 + self.scale) * (self.output_dim ** 0.5)

        elif self.normed == 'batch_norm':
            y = self.batchnorm_h(y)

        if self.bias is not None:
            y = y + self.bias

        if self.active is not None:
            y = self.active(y)

        if self.residue:
            y = y + x

        return y

class GCNLayer(nn.Module):

    def __init__(self,
               input_dim,
               output_dim,
               n_conv,
               dropout,
               residue = True,
               bias = False,
               active = F.relu):

        super(GCNLayer, self).__init__()
        self.n_conv = n_conv
        self.output_dim = output_dim

        if input_dim != output_dim:
            residue = False
        self.residue = residue

        self.layers = nn.ModuleList([
            GraphConvolution(
                input_dim,
                input_dim,
                dropout,
                residue = True,
                active= active,
            )
            for _ in range(n_conv - 1)
        ])

        self.layers.append(
            GraphConvolution(
                input_dim,
                output_dim,
                dropout,
                residue = residue,
                active=F.relu,
            )
        )

        self.scale = nn.Parameter(torch.tensor([1.500], dtype = torch.float32))


    def forward(self, inputs):

        x, adj1, adj2 = inputs
        y = x
        for _ in range(self.n_conv):
            y = self.layers[_]((y, adj1))

        N = y.shape[0]
        y_t = y - torch.mean(y, dim=0).repeat((N, 1)).view(y.shape)
        y_norm = y_t.abs().pow(2).sum(dim=1).pow(-(1 / 2)).view((N, 1))
        y = y_t * y_norm * (1 + self.scale)

        if self.residue:
            y = x + y

        return y






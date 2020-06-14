import torch
import numpy as np
import scipy.sparse as sp
from torch import nn
from math import sqrt
from torch.nn import functional as F
from utils import sparse_dropout
from config import args


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L = 2):

        super(MLPReadout, self).__init__()
        list_FC_layers = [nn.Linear(input_dim//2 ** l, input_dim//2 **(l+1), bias = False) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim// 2 **L, output_dim, bias = False))

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class GINMLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, n_Layers = 2):

        super(GINMLP, self).__init__()
        self.L = n_Layers
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.linears.append(nn.Linear(in_dim, hidden_dim, bias = False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_Layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias = False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears.append(nn.Linear(hidden_dim, out_dim, bias = False))

    def forward(self, x):

        for _ in range(self.L - 1):
            x = F.relu(self.batch_norms[_](self.linears[_](x)))

        return self.linears[self.L - 1](x)

class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim,
               dropout,
               batch_norm,
               graph_norm,
               sparse_input = True,
               bias = False,
               active = F.relu,
               residual = False,
               featureless = False):

        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.graph_norm = graph_norm
        self.sparse_input = sparse_input
        self.active = active

        self.residual = residual
        if input_dim != output_dim:
            self.residual = False

        self.featureless = featureless

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))

        self.scale = nn.Parameter(torch.tensor([0.000], dtype = torch.float32))
        nn.init.normal(self.weight)
        self.bias = None

        if bias:
            self.bias = nn.Parameters(torch.randn(output_dim))

        if batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(output_dim)

    def forward(self, inputs):

        x, adj, n_norm = inputs
        # undertaking dropout

        # solve convolution
        if not self.featureless:
            if self.sparse_input:
                prod = torch.sparse.mm(x, self.weight)
            else:
                x = x.type(torch.FloatTensor)
                prod = torch.mm(x, self.weight)
        else:
            prod = self.weight

        t_adj = adj.type(torch.sparse.DoubleTensor)
        t_prod = prod.type(torch.DoubleTensor)
        out = torch.sparse.mm(t_adj, t_prod)
        out = out.type(torch.FloatTensor)

        if self.bias is not None:
            out = out + self.bias

        if self.graph_norm:
            out = out * n_norm

        if self.batch_norm:
            if args.norm_type == 'b_norm':
                out = self.batchnorm_h(out)
            elif args.norm_type == 'sp_norm':
                y = out
                N = y.shape[0]
                y_t = y - (y.sum(0) / (N)).repeat((N, 1)).view(y.shape)
                y_norm = y_t.pow(2).sum(1).view((N, 1)).mean(0).pow(-(1 / 2))
                y_norm = y_norm.repeat((N, 1))
                y = y_t * y_norm * (1 + self.scale) * (self.output_dim ** 0.5)
                out = y

        out = self.active(out)

        if self.residual:
            out = out + x

        if self.training and self.sparse_input:
            nonzero_feat = x._nnz()
            x = sparse_dropout(x, self.dropout, nonzero_feat)
        elif self.training:
            x = F.dropout(x, self.dropout, training = self.training)

        return out

class GraphIsomorphismLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 batch_norm,
                 graph_norm,
                 residue):

        super(GraphIsomorphismLayer, self).__init__()
        self.dropout = dropout
        if in_dim != out_dim:
            residue = False

        self.residue = residue
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.weight = nn.Parameter(torch.tensor([0.000], dtype = torch.float32))
        self.MLPlayer = GINMLP(in_dim, out_dim, out_dim)

    def forward(self, inputs):

        x, adj, n_norm = inputs

        y = torch.sparse.mm(adj, x)
        y += self.weight * x

        y = self.MLPlayer(y)

        if self.graph_norm:
            y = y * n_norm

        if self.batch_norm:
            y = self.batchnorm_h(y)

        y = F.relu(y)

        if self.residue:
            y = y + x

        y = F.dropout(y, self.dropout, training = self.training)
        return y


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_feat, out_feat, dropout, alpha, concat=True):
        '''
        :param in_feat:  input feature dimension
        :param out_feat:  out feature dimension
        :param dropout:   dropout rate of the Layer
        :param alpha:     alpha for the leaky relu
        :param concat: whether concatenate or not
        '''
        super(GraphAttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # calculate WH = (Wh1, Wh2, Wh3, ...Whn)
        self.W = nn.Parameter(torch.zeros(size=(in_feat, out_feat)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(out_feat * 2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        # using the adj matrix and the feature x
        x, support = input

        # calculating H  = X * W
        h = torch.mm(x, self.W)
        N = h.size()[0]

        # calculating attention matrix
        # which is really tricky, using tensor techniques

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)]).view(N, -1, 2 * self.out_feat)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        alter = -9e15 * torch.ones_like(e)
        out = torch.where(support > 0, e, alter)
        out = F.softmax(out, dim=1)
        out = F.dropout(out, self.dropout, training=self.training)

        h_out = torch.matmul(out, h)

        if self.concat:
            return F.elu(h_out)
        else:
            return h_out


class SpecialSpmmFunction(torch.autograd.Function):
    """
    special function for sparse layer backpropagation
    """

    @staticmethod
    def forward(ctx, indices, values, shape, b):

        assert indices.requires_grad == False
        # convert the (indices, values, shape) the features into sparse
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]

        return torch.matmul(a, b)

    '''
    using the derivation for the matrix

    A * B = out_put
    dloss/dA = dloss/dout_put * B.transpose
    dloss/dB = A.transpose * dloss/dout_put
    '''

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class Specialspmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


'''
Attention Head Layer 
'''
class SpAttentionLayer(nn.Module):

    def __init__(self, in_feat, out_feat, dropout, alpha, concat=True):

        super(SpAttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_feat, out_feat)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_feat)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.special_spmm = Specialspmm()

    def forward(self, input):

        x, support = input

        device = x.device

        N = x.size()[0]

        edge = support._indices()
        h = torch.mm(x, self.W)

        assert not torch.isnan(h).any()

        edge_h = torch.cat([h[edge[0, :], :], h[edge[1, :], :]], dim=1).t()

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]),
                                     torch.ones(size=(N, 1)).to(device))

        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)

        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)

        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GATHeadLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, graph_norm, batch_norm):

        super(GATHeadLayer, self).__init__()

        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.fc = nn.Linear(input_dim, output_dim, bias = False)
        nn.init.zeros_(self.fc.weight)

        self.attn_fc = nn.Linear(2 * output_dim, 1, bias = False)
        nn.init.zeros_(self.attn_fc.weight)

        self.leaky_relu = nn.LeakyReLU(alpha)

        self.batchnorm_h = nn.BatchNorm1d(output_dim)
        self.special_spmm = Specialspmm()

    def forward(self, input):

        x, support, n_norm = input

        device = x.device

        #edge_attention
        N = x.size()[0]
        edge = support._indices()
        #h = torch.mm(x, self.W)

        h = self.fc(x)

        assert not torch.isnan(h).any()

        edge_h = torch.cat([h[edge[0, :], :], h[edge[1, :], :]], dim=1)

        edge_e = torch.exp(-self.leaky_relu(self.attn_fc(edge_h).squeeze()))

        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]),
                                     torch.ones(size=(N, 1)).to(device))

        #edge_e = F.dropout(edge_e, self.dropout, training = self.training)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)

        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)

        assert not torch.isnan(h_prime).any()

        if self.graph_norm:
            h_prime = h_prime * n_norm

        if self.batch_norm:
            h_prime = self.batchnorm_h(h_prime)

        h_prime = F.elu(h_prime)
        h_prime = F.dropout(h_prime, self.dropout, training = self.training)

        return h_prime



class GATLayer(nn.Module):

    def __init__(self, input_dim, output_dim, n_heads,
                 alpha, dropout, graph_norm, batch_norm,
                 residue = False, activation = None):

        super(GATLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        self.residue = residue


        if (input_dim != output_dim * n_heads):
            self.residue = False

        self.attentions = nn.ModuleList()
        for _ in range(n_heads):
            self.attentions.append(GATHeadLayer(input_dim, output_dim,
                                                dropout, alpha,
                                                graph_norm, batch_norm))

    def forward(self, inputs):

        x, support, n_norm = inputs

        out = x

        atten_res = [atten((out, support, n_norm)) for atten in self.attentions]
        out = torch.cat(atten_res, dim = 1)

        if self.residue:
            out = out + x

        return out



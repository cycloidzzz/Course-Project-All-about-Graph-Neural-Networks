import torch
from torch import nn
from torch.nn import functional as F
from utils import sparse_dropout


class MLPReadout(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout, L = 3):

        super(MLPReadout, self).__init__()
        self.n_layer = L
        self.dropout = dropout

        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hid_dim, bias = False)
        ])

        for i in range(L - 2):
            self.layers.append(nn.Linear(hid_dim//(2 ** i), hid_dim//(2 ** (i + 1)), bias = False))

        self.layers.append(nn.Linear(hid_dim// (2 ** (L - 2)), out_dim, bias = False))

    def forward(self, x):

        for _ in range(self.n_layer):
            x = self.layers[_](x)
            x = F.relu(x)

        return x

class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim, nonzero_feat,
               dropout,
               sparse_input = True,
               bias = False,
               active = F.relu,
               featureless = False):
        super(GraphConvolution, self).__init__()

        self.nonzero_feat = nonzero_feat
        self.dropout = dropout
        self.sparse_input = sparse_input
        self.active = active
        self.featureless = featureless

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        nn.init.kaiming_normal(self.weight)

        self.bias = None

        if bias:
            self.bias = nn.Parameters(torch.randn(output_dim))

    def forward(self, inputs):

        x, adj = inputs
        # undertaking dropout
        if self.training and self.sparse_input:
            x = sparse_dropout(x, self.dropout, self.nonzero_feat)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # solve convolution
        if not self.featureless:
            if self.sparse_input:
                x = x.type(torch.sparse.FloatTensor)
                prod = torch.sparse.mm(x, self.weight)
            else:
                prod = torch.mm(x, self.weight)
        else:
            prod = self.weight

        t_adj = adj.type(torch.sparse.DoubleTensor)
        t_prod = prod.type(torch.DoubleTensor)
        out = torch.sparse.mm(t_adj, t_prod)
        out = out.type(torch.FloatTensor)

        if self.bias is not None:
            out = out + self.bias

        return (self.active(out), adj)


class GraphIsomorphismLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout):

        super(GraphIsomorphismLayer, self).__init__()
        self.dropout = dropout

        self.weight = nn.Parameter(torch.tensor([3.000], dtype = torch.float32))
        self.MLPLayer = MLPReadout(in_dim, 64, out_dim, dropout)

        self.residue = False
        if in_dim == out_dim:
            self.residue = True

    def forward(self, inputs):

        x, adj = inputs

        y = torch.sparse.mm(adj, x)

        y = y + self.weight * x

        y = F.dropout(y, self.dropout, training = self.training)

        y =  self.MLPLayer(y)

        if self.residue:
            y = y + x

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

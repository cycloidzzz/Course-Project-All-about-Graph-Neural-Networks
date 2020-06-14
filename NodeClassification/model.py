import torch
from torch import nn
from torch.nn import functional as F
from layer import GraphConvolution, GraphIsomorphismLayer, GraphAttentionLayer, SpAttentionLayer
from config import args


class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, nonzero_feature):

        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


        self.cov1 = GraphConvolution(
            input_dim, args.hidden, nonzero_feature,
            args.dropout,
            sparse_input = True,
            active = torch.relu
        )

        self.cov2 = GraphConvolution(
            args.hidden, output_dim, nonzero_feature,
            args.dropout,
            sparse_input = False,
            active = torch.relu
        )


    def forward(self, inputs):

        x, A = inputs
        out1 = self.cov1((x, A))
        out1 = out1[0]
        out2 = self.cov2((out1, A))

        return out2[0]


class GIN(nn.Module):

    def __init__(self, in_dim, hid_dim, n_class, dropout):

        super(GIN, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(in_dim, hid_dim, bias = False)
        self.fc2 = nn.Linear(3 * hid_dim, n_class, bias = False)
        self.layer1 = GraphIsomorphismLayer(hid_dim, hid_dim, dropout)
        self.layer2 = GraphIsomorphismLayer(hid_dim, hid_dim, dropout)

    def forward(self, inputs):

        x, adj = inputs


        feats = []
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.fc1(x)
        feats.append(x)
        x = self.layer1((x, adj))
        feats.append(x)
        x = self.layer2((x, adj))
        feats.append(x)
        x = self.fc2(torch.cat(feats, dim = 1))

        return x


class GAT(nn.Module):

    def __init__(self, in_dim, hid_dim, nclass, dropout, alpha, nheads):
        '''
        :param in_dim: dimension of input feature
        :param hid_dim: hidden layer dimension
        :param nclass: classification classes
        :param dropout:
        :param alpha: for leakyrelu
        :param nheads: nheads for GAT
        '''
        super(GAT, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.nclass = nclass
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_dim, hid_dim, dropout, alpha, concat = True) for i in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention{}'.format(i), attention)

        self.out_attention = GraphAttentionLayer(hid_dim * nheads, nclass, dropout, alpha, concat = False)

    def forward(self, input):
        '''

        :param input: a tuple (feature, adjacency)
        :return: classification result out
        '''
        x, support = input
        # 1 layer
        x = F.dropout(x, self.dropout, training = self.training)
        out = torch.cat([atten((x, support)) for atten in self.attentions], dim = 1)
        # 3 layer
        out = F.dropout(out, self.dropout, training = self.training)
        out = F.elu(self.out_attention((out, support)))
        return F.log_softmax(out, dim = 1)


class spGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, nclass, dropout, alpha, nheads):
        super(spGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpAttentionLayer(in_dim,
                                            hid_dim,
                                            dropout,
                                            alpha,
                                            concat = True) for i in range(nheads)]

        for i, atten in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), atten)

        self.out_attention = SpAttentionLayer(hid_dim * nheads,
                                         nclass,
                                         dropout,
                                         alpha,
                                         concat = False)


    def forward(self, input):

        x, support = input
        x = F.dropout(x, self.dropout, training = self.training)
        out = torch.cat([atten((x, support)) for atten in self.attentions], dim= 1)
        out = F.dropout(out, self.dropout, training = self.training)
        out = F.elu(self.out_attention((out, support)))
        return F.log_softmax(out, dim = 1)
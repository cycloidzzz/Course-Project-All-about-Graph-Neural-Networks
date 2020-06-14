import torch
from torch import nn
from torch.nn import functional as F
from layer import GraphIsomorphismLayer, GraphConvolution, GraphAttentionLayer, SpAttentionLayer, GATLayer, MLPReadout
from config import args

def meanpooling(y, indicator, index):
    sz = y.shape[1]
    return torch.cat([(y[mask].sum(0) * 1.0/mask.sum(0)).view(1, sz) for mask in
                      [(indicator == x) for x in index]], dim = 0)


def maxpooling(y, indicator, index):
    return torch.cat([y[mask].max(0) for mask in [(indicator == x) for x in index]], dim = 0)

def sumpooling(y, indicator, index):
    feat_size = y.shape[1]
    return torch.cat([y[mask].sum(0).view(1, feat_size)
                      for mask in [(indicator == x) for x in index]], dim = 0)


class GCN(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_class,
                 in_feat_dropout,
                 n_layer,
                 batch_norm,
                 graph_norm,
                 residual,
                 embedding = False):

        super(GCN, self).__init__()
        self.n_layer = n_layer
        self.Embedding = embedding
        self.in_feat_dropout = in_feat_dropout


        if self.Embedding:
            self.embed1 = nn.Embedding(input_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphConvolution(hidden_dim,
                                                      hidden_dim,
                                                      args.dropout,
                                                      batch_norm,
                                                      graph_norm,
                                                      residual = residual,
                                                      sparse_input = False,
                                                      active = F.relu)
                                     for l in range(n_layer - 1)])

        self.layers.append(GraphConvolution(hidden_dim,
                                            output_dim,
                                            args.dropout,
                                            batch_norm,
                                            graph_norm,
                                            residual = residual,
                                            sparse_input = False,
                                            active = F.relu))

        self.MLP_layer = MLPReadout(output_dim, n_class)


    def forward(self, inputs):

        x, A, indicator, n_norm, graph_index = inputs
        if self.Embedding == True:
            x = x.type(torch.LongTensor)
            out = self.embed1(x)
        else:
            out = x.type(torch.FloatTensor)
            out = self.fc1(out)
            out = F.dropout(out, self.in_feat_dropout, training = self.training)

        y = out
        for l in range(self.n_layer):
            y = self.layers[l]((y, A, n_norm))

        y = meanpooling(y, indicator, graph_index)

        return self.MLP_layer(y)

    #L2 Regularization
    def L2_loss(self):

        layer = self.layers.children()
        tmp = iter(layer)

        loss = None
        for layer in tmp:
            for p in layer.parameters():

                if loss is None:
                    loss = p.pow(2).sum()
                else:
                    loss += p.pow(2).sum()

        return loss


class GIN(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 n_class,
                 dropout,
                 n_layers,
                 batch_norm,
                 graph_norm,
                 residue,
                 embedding):

        super(GIN, self).__init__()
        self.n_layers = n_layers
        self.embedding = embedding

        if embedding:
            self.embed = nn.Embedding(in_dim, hid_dim)
        else:
            self.fc1 = nn.Linear(in_dim, hid_dim)

        self.feat_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GraphIsomorphismLayer(
                hid_dim,
                hid_dim,
                dropout,
                batch_norm,
                graph_norm,
                residue)

            for _  in range(n_layers)])

        self.linear_score = nn.ModuleList([
            nn.Linear(hid_dim, n_class)
            for _ in range(1 + n_layers)])



    def forward(self, inputs):

        x, A, indicator, n_norm, graph_index = inputs

        if self.embedding:
            x = self.embed(x)
        else:
            x = self.fc1(x)

        x = self.feat_dropout(x)

        feat_lst = [x]
        for _  in range(self.n_layers):
            x = self.layers[_]((x, A, n_norm))
            feat_lst.append(x)


        score_over_layer = 0
        for i, feat in enumerate(feat_lst):

            feat = sumpooling(feat, indicator, graph_index)
            tmp = self.linear_score[i](feat)
            score_over_layer += tmp

        return score_over_layer



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

        self.attention1 = [GraphAttentionLayer(in_dim, hid_dim, dropout, alpha, concat = True) for i in range(nheads)]
        for i, attention in enumerate(self.attention1):
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
        out = torch.cat([atten((x, support)) for atten in self.attention1], dim = 1)
        # 3 layer
        out = F.dropout(out, self.dropout, training = self.training)
        out = F.elu(self.out_attention((out, support)))
        return F.log_softmax(out, dim = 1)



class spGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_class, n_heads, n_layers,
                 dropout, alpha, graph_norm, batch_norm, residue, embedding):
        super(spGAT, self).__init__()

        self.n_layers = n_layers

        self.embedding = embedding
        if embedding:
            self.embedding = nn.Embedding(input_dim, hidden_dim * n_heads)
        else:
            self.embedding = nn.Linear(input_dim, hidden_dim * n_heads)

        self.in_feature_dropout = nn.Dropout(dropout)


        self.layers = nn.ModuleList([GATLayer(hidden_dim * n_heads, hidden_dim,
                                              n_heads, alpha, dropout,
                                              graph_norm, batch_norm, residue) for _ in range(n_layers-1)])

        self.layers.append(GATLayer(hidden_dim * n_heads, output_dim, 1, alpha, dropout, graph_norm, batch_norm, residue))

        self.MLP_layer = MLPReadout(output_dim, n_class)


    def forward(self, inputs):
        x, adj, indicator, n_norm, graph_index = inputs

        out = self.embedding(x).to(x.device)

        #print("this is out", out)
        out = self.in_feature_dropout(out)

        for _ in range(self.n_layers):
            out = self.layers[_]((out, adj, n_norm))

        out = meanpooling(out, indicator, graph_index)

        out = self.MLP_layer(out)

        return out



import torch
from torch import nn
from torch.nn import functional as F
from layer import GCNLayer, GraphConvolution
from config import args


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_class, n_layers):

        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #nn.init.xavier_normal(self.fc1.weight, gain = 1.414)

        self.layers = nn.ModuleList([
            GraphConvolution(
                hidden_dim,
                hidden_dim,
                args.dropout,
                residue = True,
                active = F.relu,
                normed = args.norm_type
            )
            for _ in range(n_layers - 1)
        ])

        self.layers.append(
            GraphConvolution(
                hidden_dim,
                n_class,
                args.dropout,
                residue = False,
                active = F.relu,
                normed = args.norm_type
            )
        )

        #self.fc2 = nn.Linear(output_dim, n_class)
        #nn.init.xavier_normal(self.fc2.weight, gain = 1.414)

    def forward(self, inputs):

        x, adj1, adj2 = inputs
        x = self.fc1(x)
        x = F.dropout(x, args.dropout, training = self.training)

        for i in range(self.n_layers):
            #print("in layer %d"%(i))
            x = self.layers[i]((x, adj1, adj2))

        #x = self.fc2(x)

        return x

    #L2 Regularization
    def L2_loss(self):

        loss = None
        for layer in self.layers:
            for p in layer.parameters():

                if loss is None:
                    loss = p.pow(2).sum()
                else:
                    loss += p.pow(2).sum()

        return loss


#implementation of dense net
class DenseGCN(nn.Module):

    def __init__(self, n_layers, input_init_dim, hidden_dim, growth_rate, out_dim, n_class):

        #self.dropout = dropout
        super(DenseGCN, self).__init__()
        self.n_layers = n_layers

        self.fc1 = nn.Linear(input_init_dim, hidden_dim)

        layers = []
        num_features = hidden_dim

        for i in range(n_layers - 1):

            layers.append(GraphConvolution(
                num_features,
                growth_rate,
                args.dropout,
                residue = False,
                active = F.relu,
                normed = True
            ))

            num_features += growth_rate

        layers.append(GraphConvolution(
            num_features,
            out_dim,
            args.dropout,
            residue = False,
            active = F.relu,
            normed = True
        ))

        self.layers = nn.ModuleList(layers)
        self.fc2 = nn.Linear(out_dim, n_class)

    def forward(self, inputs):

        x, adj1, adj2 = inputs
        x = F.dropout(x, args.dropout, training = self.training)
        x = self.fc1(x)

        outs = [x]

        for i in range(self.n_layers):
            tmp_in = torch.cat(outs, dim = 1)
            tmp_out = self.layers[i]((tmp_in, adj1))
            outs.append(tmp_out)

        return self.fc2(outs[-1])


import os
import glob
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
import itertools
from data import get_dataset, normalized_feature, normalized_adj, sparse_to_tuple,  tuple_to_tensor
from model import GIN, GCN, GAT, spGAT
from config import args
from layer import MLPReadout
from utils import masked_loss, masked_accuracy, accuracy
from datasets import get_data, mydata_loader, array2tensor


seed = 41
np.random.seed(seed)
torch.random.manual_seed(seed)

dataset = get_data(data_name = args.dataset)

# loading the training arguments
device = torch.device('cpu')
batch_size = args.batch_size


embedding = True if args.dataset == 'DD' else False


feat_dim = max(dataset.node_labels) + 1 if args.dataset == "DD" else dataset.features.shape[1]
hidden_dim = args.hidden
output_dim = 128
n_class = max(dataset.graph_labels) + 1


print("feature dim:", feat_dim)


if args.model == "gcn":
    net = GCN(feat_dim,
              hidden_dim,
              output_dim,
              n_class,
              args.dropout,
              args.layers,
              True,
              True,
              True,
              embedding = embedding)
elif args.model == 'gat':
    net = spGAT(feat_dim, hidden_dim, output_dim, n_class,
                8,
                args.layers,
                args.dropout,
                0.01,
                True,
                True,
                True,
                embedding)

elif args.model == 'gin':
    net = GIN(feat_dim,
              hidden_dim,
              output_dim,
              n_class,
              args.dropout,
              args.layers,
              True,
              True,
              True,
              embedding)

net.to(device)
optimizer = optim.Adam(net.parameters(),
                       lr = args.learning_rate,
                       weight_decay = args.weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=25,
                                                 verbose = True)

min_lr = 1e-7

train_index, val_index, test_index = dataset.train_index, \
                                     dataset.val_index, \
                                     dataset.test_index

loss_train = []
acc_valid = []


def train(dataset, train_index):

    train_acc, train_loss, train_size = 0.0, 0.0, len(train_index)

    net.train()
    for (adj, features, indicator, labels, n_norm, batch_idx) in mydata_loader(batch_size,
                                                                               dataset,
                                                                               train_index):

        tmp_size = len(labels)
        #out = torch.cat([net((x[1], x[0])).view(1, output_dim) for x in lst], dim=0).to(device)

        out = net((features, adj, indicator, n_norm, batch_idx))
        optimizer.zero_grad()
        loser = nn.CrossEntropyLoss()
        loss = loser(out, labels)


        #grad = torch.autograd.grad(loss, net.parameters())
        #print(grad)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss += loss.detach().item() * tmp_size
        train_acc += accuracy(out, labels)

    train_acc /= train_size
    train_loss /= train_size

    loss_train.append(train_loss)

    return train_acc, train_loss

#mini-batch calculations for validation/test set
def evalution_network(dataset, index):

    val_acc, val_loss, tot_size = 0.0, 0.0, len(index)
    net.eval()

    with torch.no_grad():
        for (adj, features, indicator, labels, n_norm, batch_idx) in mydata_loader(batch_size,
                                                                                   dataset,
                                                                                   index):

            tmp_size = len(labels)

            out = net((features, adj, indicator, n_norm, batch_idx))
            val_acc += accuracy(out, labels)

            loser = nn.CrossEntropyLoss()
            _loss = loser(out, labels)

            val_loss += _loss.item() * tmp_size

    val_acc /= tot_size
    val_loss /= tot_size

    return val_acc, val_loss


loss_val = []
best_epoch = 0
bad_cnt = 0
best_loss = args.epoches + 1


if args.method == 'mini-batch':
    for epoch in range(args.epoches):
        #loss_val.append(train(epoch))

        train_acc, train_loss = train(dataset, train_index)
        _acc, _loss = evalution_network(dataset, val_index)

        loss_val.append(_loss)
        acc_valid.append(_acc.item())

        print("in epoch {:04d}".format(epoch),
              "train accuracy  {:.6f}".format(train_acc),
              "train loss {:.6f}".format(train_loss),
              "val accuracy {:.6f}".format(_acc),
              "val loss {:.6f}".format(_loss))

        torch.save(net.state_dict(), '{}.pkl'.format(epoch))
        scheduler.step(loss_val[-1])

        if optimizer.param_groups[0]['lr'] < min_lr:
            print("Learning rate equals to min_lr ")
            break

        if loss_val[-1] < best_loss:
            best_epoch = epoch
            best_loss = loss_val[-1]
            bad_cnt = 0
        else:
            bad_cnt += 1

        if bad_cnt > args.early_stopping:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)


file = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)


net.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

test_acc, test_loss = evalution_network(dataset, test_index)
print("test accuracy {:.6f}".format(test_acc),
      "test loss {:.6f}".format(test_loss))



def data_printer(lst, filename):

    file = open(filename, "w")
    for number in lst:
        file.write(str(number) + "\n")
    file.close()

filename = "{}layers_{}_{}_{}_train_loss.txt".format(args.layers, args.norm_type, args.model, args.dataset)
print("len is ", len(loss_train))
data_printer(loss_train, filename)

filename = "{}layers_{}_{}_{}_valid_acc.txt".format(args.layers, args.norm_type, args.model, args.dataset)
print("val len is ", len(acc_valid))
data_printer(acc_valid, filename)





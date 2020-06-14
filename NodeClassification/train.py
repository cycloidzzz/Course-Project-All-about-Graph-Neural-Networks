import torch
import glob
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
import scipy.sparse as sp
from data import get_dataset, normalized_feature, normalized_adj, sparse_to_tuple, data_iter, tuple_to_tensor
from model import GCN, GIN, GAT, spGAT
from config import args
from utils import masked_loss, masked_accuracy, accuracy, data_printer


seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)

support, feature, y_train, y_val, y_test, \
train_mask, val_mask, test_mask, \
train_idx, val_idx, test_idx = get_dataset(args.dataset)

#cpu/cuda
device = torch.device('cpu')

# convert all numpy arrays to torch tensors
train_label = torch.from_numpy(y_train).long().to(device)
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim = 1)

train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device)

val_label = torch.from_numpy(y_val).long().to(device)
val_label = val_label.argmax(dim = 1)
val_mask  = torch.from_numpy(val_mask.astype(np.int)).to(device)

test_label = torch.from_numpy(y_test).long().to(device)
test_label = test_label.argmax(dim = 1)
test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)


tmp_feat = normalized_feature(feature)

if args.model == 'gcn':
    i = torch.from_numpy(tmp_feat[0]).long().to(device)
    v = torch.from_numpy(tmp_feat[1]).to(device)
    tmp_feat = torch.sparse.FloatTensor(i.t(), v, tmp_feat[2]).to(device)
elif args.model != 'gcn':
    row = np.array(tmp_feat[0][:, 0])
    col = np.array(tmp_feat[0][:, 1])
    print("this is feature number", tmp_feat[2])
    tmp_feat = sp.coo_matrix((tmp_feat[1], (row, col)), tmp_feat[2])
    tmp_feat = torch.FloatTensor(np.array(tmp_feat.todense()))

if args.model != 'gin':
    tmp_adj = normalized_adj(support)
else:
    tmp_adj = sparse_to_tuple(support.astype(np.float32))

i = torch.from_numpy(tmp_adj[0]).long().to(device)
v = torch.from_numpy(tmp_adj[1]).to(device)
tmp_adj = torch.sparse.FloatTensor(i.t(), v, tmp_adj[2]).to(device)


train_idx = torch.LongTensor(train_idx)
val_idx = torch.LongTensor(val_idx)
test_idx = torch.LongTensor(test_idx)


# loading the training arguments

feat_dim = feature.shape[1]
batch_size = args.batch_size

if args.model == 'gcn':
    num_features_nonzero =  tmp_feat._nnz()
    net = GCN(feat_dim, num_classes, num_features_nonzero)
    net.to(device)
elif args.model == 'gat':
    net = spGAT(
        feat_dim,
        args.hidden, #8
        num_classes,
        args.dropout,
        args.alpha,
        args.nheads)
elif args.model == 'gin':
    net = GIN(
        feat_dim,
        args.hidden,
        num_classes,
        args.dropout)


optimizer = optim.Adam(net.parameters(),
                       lr = args.learning_rate,
                       weight_decay = args.weight_decay)

train_loss = []
valid_acc = []

if args.model == 'gcn' or args.model == 'gin':
    net.train()
    tmp_feat, tmp_adj, train_label, train_mask = Variable(tmp_feat), Variable(tmp_adj), Variable(
            train_label), Variable(train_mask)
    for epoch in range(args.epoches):

        net.train()
        out = net((tmp_feat, tmp_adj))
        loss = masked_loss(out, train_label, train_mask, type='aha')
        acc = masked_accuracy(out, train_label, train_mask, type='aha')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        net.eval()
        out = net((tmp_feat, tmp_adj))
        val_acc = masked_accuracy(out, val_label, val_mask, type = 'aha')

        valid_acc.append(val_acc.item())


        if epoch % 10 == 0:
            print("%d %.10f %.10f" % (epoch, loss.item(), val_acc.item()))

    # for test set
    net.eval()
    out = net((tmp_feat, tmp_adj))

    acc = masked_accuracy(out, test_label, test_mask, type='aha')
    print('test: %.10f' % (acc.item()))


elif args.model == 'gat':
    validation = True
    def train(epoch):

        net.train()
        optimizer.zero_grad()
        out = net((tmp_feat, tmp_adj))
        loss = F.nll_loss(out[train_idx], train_label)
        acc_train = accuracy(out[train_idx], train_label)

        loss.backward()
        optimizer.step()

        if validation:
            net.eval()
            out = net((tmp_feat, tmp_adj))

        loss_val = F.nll_loss(out[val_idx], val_label)
        acc_val = accuracy(out[val_idx], val_label)

        print('epoch: {}'.format(epoch),
              'train_loss: {:.5f}'.format(loss.item()),
              'train_accuracy: {:.5f}'.format(acc_train.item()),
              'val_loss: {:.5f}'.format(loss_val.item()),
              'val_accuracy: {:.5f}'.format(acc_val.item()))

        train_loss.append(loss.item())
        valid_acc.append(acc_val.item())

        return loss_val.data.item()

    loss_values = []
    best_epoch = 0
    best = args.epoches + 1
    bad_cnt = 0
    for epoch in range(args.epoches):

        loss_values.append(train(epoch))
        torch.save(net.state_dict(), '{}.pkl'.format(epoch))
        
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_cnt = 0
        else:
            bad_cnt += 1

        if bad_cnt == args.patience:
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

    net.eval()
    out = net((tmp_feat, tmp_adj))
    loss = F.nll_loss(out[test_idx], test_label)
    acc = accuracy(out[test_idx], test_label)
    print("test : loss %.10f, accuracy %.10f" % (loss, acc))


filename = "{}_{}_train_loss.txt".format(args.model, args.dataset)
print("len is ", len(train_loss))
data_printer(train_loss, filename)

filename = "{}_{}_valid_acc.txt".format(args.model, args.dataset)
print("val len is ", len(valid_acc))
data_printer(valid_acc, filename)
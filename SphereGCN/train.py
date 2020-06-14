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
from model import GCN, DenseGCN
from config import args
from utils import masked_loss, masked_accuracy, accuracy


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
print(train_label.shape[1])
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim = 1)

train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device)

val_label = torch.from_numpy(y_val).long().to(device)
val_label = val_label.argmax(dim = 1)
val_mask  = torch.from_numpy(val_mask.astype(np.int)).to(device)

test_label = torch.from_numpy(y_test).long().to(device)
test_label = test_label.argmax(dim = 1)
test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)

#feature = sparse_to_tuple(feature)
#support = sparse_to_tuple(support)
#convert feature and adjacency-matrix to sparse torch tensors
tmp_feat = normalized_feature(feature)


row = np.array(tmp_feat[0][:, 0])
col = np.array(tmp_feat[0][:, 1])
tmp_feat = sp.coo_matrix((tmp_feat[1], (row, col)), tmp_feat[2])
tmp_feat = torch.FloatTensor(np.array(tmp_feat.todense()))

adj_lst = normalized_adj(support)
adj_tensor = []
for i in range(len(adj_lst)):
    tmp = adj_lst[i]
    idx = torch.from_numpy(tmp[0]).long().to(device)
    v = torch.from_numpy(tmp[1].astype(np.float32)).to(device)
    tmp = torch.sparse.FloatTensor(idx.t(), v, tmp[2]).to(device)
    adj_tensor.append(tmp)


tmp_adj = adj_tensor[0]


train_idx = torch.LongTensor(train_idx)
val_idx = torch.LongTensor(val_idx)
test_idx = torch.LongTensor(test_idx)


# loading the training arguments

feat_dim = feature.shape[1]
batch_size = args.batch_size

if args.model == 'gcn':
    net = GCN(feat_dim, args.hidden, 14, num_classes, args.layers)
    net.to(device)
elif args.model == 'densegcn':
    net = DenseGCN(15, feat_dim, 28, 24, 24, num_classes)
    net.to(device)


optimizer = optim.Adam(net.parameters(),
                       lr = args.learning_rate,
                       weight_decay = args.weight_decay)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode = 'min',
                                               factor = 0.6,
                                               patience = 30,
                                               verbose = True)


min_lr = 1e-6

loss_train = []
acc_valid = []



if args.model == 'gcn' or args.model == 'densegcn':
    net.train()

    tmp_feat, train_label, train_mask = Variable(tmp_feat), Variable(
        train_label), Variable(train_mask)
    def train_x():
        net.train()
        out = net((tmp_feat, adj_tensor[0], adj_tensor[1]))
        loss = masked_loss(out, train_label, train_mask, type='aha')
        acc = masked_accuracy(out, train_label, train_mask, type='aha')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), acc.item()

    def evalution(mask, label):
        #for test and evaluation
        net.eval()
        out  = net((tmp_feat, adj_tensor[0], adj_tensor[1]))
        loss = masked_loss(out, label, mask, type = 'aha')
        acc = masked_accuracy(out, label, mask, type='aha')

        return loss.item(), acc.item()

    loss_values = []

    best_epoch = 0
    best = args.epoches + 1
    bad_cnt = 0

    for epoch in range(args.epoches):

        train_loss, train_acc = train_x()
        loss_train.append(train_loss)

        val_loss, val_acc = evalution(val_mask, val_label)

        scheduler.step(val_loss)

        if optimizer.param_groups[0]['lr'] < min_lr:
            print("lr reduced to min lr ")
            break
            
        print("in epoch %d, train loss %.10f, train accuracy %.10f"%(epoch, train_loss, train_acc))
        print("validation loss %.10f, validation accuracy %.10f"%(val_loss, val_acc))

        loss_values.append(val_loss)
        acc_valid.append(val_acc)

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

    test_loss, test_acc = evalution(test_mask, test_label)
    print("the layer is", args.layers)
    print("on test: loss %.10f, accuracy %.10f"%(test_loss, test_acc))
    print(loss_train)
    print(acc_valid)



def data_printer(lst, filename):
    file = open(filename, "w")
    for number in lst:
        file.write(str(number) + "\n")
    file.close()

print("this is {} gcn with {} layers".format(args.norm_type, args.layers))
filename = "{}_{}_{}_{}_train_loss.txt".format(args.norm_type, str(args.layers), args.model, args.dataset)
print("len is ", len(loss_train))
data_printer(loss_train, filename)

filename = "{}_{}_{}_{}_valid_acc.txt".format(args.norm_type, str(args.layers), args.model, args.dataset)
print("val len is ", len(acc_valid))
data_printer(acc_valid, filename)

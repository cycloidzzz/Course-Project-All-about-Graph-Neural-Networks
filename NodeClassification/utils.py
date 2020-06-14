'''
adding some useful functions
'''

import torch
from torch import nn
from torch.nn import functional as F

def masked_loss(out, label, mask, type = 'aha'):
    loss = F.cross_entropy(out, label, reduction = 'none')
    if type != 'mini-batch':
        mask = mask.float()
        mask = mask/mask.mean()
        loss *= mask

    loss = loss.mean()
    return loss

def masked_accuracy(out, label, mask, type = 'mini-batch'):

    pred = out.argmax(dim = 1)
    acc = torch.eq(pred, label).float()
    if type != 'mini-batch':
        mask = mask.float()
        mask = mask / mask.mean()
        acc *= mask

    acc = acc.mean()
    return acc

def sparse_dropout(x, rate, noise_shape):


    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()

    idx = x._indices()
    val = x._values()

    idx = idx[:, dropout_mask]
    val = val[dropout_mask]

    out = torch.sparse.FloatTensor(idx, val, x.shape).to(x.device)

    out = out * (1./(1 - rate))
    return out


def accuracy(out, labels):
    pred = out.max(1)[1].type_as(labels)
    acc = pred.eq(labels).double()
    acc = acc.sum()
    return acc * 1.0/len(labels)


def data_printer(lst, filename):

    file = open(filename, "w")
    for number in lst:
        file.write(str(number) + "\n")
    file.close()
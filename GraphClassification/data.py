import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def parse_index_file(filename):
    """
    :param filename:
    :return:
    """
    idx = []
    with open(filename) as f:
        for line in f:
            idx.append(int(line.strip('\n')))
    return idx

def gen_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype = bool)

def tuple_to_tensor(sparse, device):
    i = torch.from_numpy(sparse[0]).long().to(device)
    v = torch.from_numpy(sparse[1]).to(device)
    return torch.sparse.tensor(i.t(), v, sparse[2]).to(device)

def get_dataset(dataset_str):
    """
    load inputs from gcn/data

    ind.dataset.x
    ind.dataset.tx
    ind.dataset.allx
    ind.dataset.y
    ind.dataset.ty
    ind.dataset.ally

    ind.dataset.graph

    ind.test.index
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    object = []
    for name in names:
        with open("data/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3,0):
                object.append(pkl.load(f, encoding = 'latin1'))
            else:
                object.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(object)
    test_idx = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_reorder = np.sort(test_idx)


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_reorder, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_reorder, :]

    idx_test = test_idx_reorder.tolist()
    #train
    idx_train = range(len(y))
    #validation
    idx_val = range(len(y), len(y) + 500)


    train_mask = gen_mask(idx_train, labels.shape[0])
    test_mask = gen_mask(idx_test, labels.shape[0])
    val_mask = gen_mask(idx_val, labels.shape[0])


    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, : ] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """

    :param mx: in shape of coo_matrix,
    :return: ((row, col), data, shape)
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])

    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalized_feature(features):
    """
    :param features: an lil matrix
    :return: Renormalized feature matrix and return to tuple representation
    """

    rowsums = np.array(features.sum(1))
    r_rev = np.power(rowsums, -1).flatten()
    r_rev[np.isinf(r_rev)] = 0
    r_mat = sp.diags(r_rev)
    features = r_mat.dot(features)

    return sparse_to_tuple(features)


def get_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_rev = np.power(rowsum, -0.5).flatten()
    d_rev[np.isinf(d_rev)] = 0
    d_mat = sp.diags(d_rev)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()

def normalized_adj(adj):
    normalized = get_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(normalized)

#for mini-batch iter
def data_iter(batch_size, adj, features, labels, train_mask):
    '''
    :param batch_size:
    :param adj: scipy lil_matrix
    :param features: scipy lil_matrix
    :param labels: torch tensor
    :param train_mask: mask for training set, as numpy array, return a LongTensor
    :return:
    '''
    device = labels.device
    train_idx = np.nonzero(train_mask)[0]
    np.random.shuffle(train_idx)
    mx, mask_len = len(train_idx), len(train_mask)
    for i in range(0, mx, batch_size):
        # to select the index for mini-batch
        tmp_idx = train_idx[i : min(i + batch_size, mx)]

        j = torch.LongTensor(tmp_idx)

        # create mini-batch adjacency matrix
        support = adj[tmp_idx][: , tmp_idx]
        support = normalized_adj(support)
        index = torch.from_numpy(support[0]).long().to(device)
        data = torch.from_numpy(support[1]).to(device)
        support = torch.sparse.FloatTensor(index.t(), data, support[2]).to(device)

        # create mini-batch features matrix
        feat = features[tmp_idx]
        feat = normalized_feature(feat)
        index = torch.from_numpy(feat[0]).long().to(device)
        data = torch.from_numpy(feat[1]).to(device)
        feat = torch.sparse.FloatTensor(index.t(), data, feat[2]).to(device)

        # create mask for the mini-batch
        res_mask = np.zeros(mask_len)
        res_mask[tmp_idx] = 1
        res_mask = torch.from_numpy(res_mask.astype(np.int)).to(device)


        yield   support, feat,  \
                labels.index_select(0, j).to(device), \
                res_mask


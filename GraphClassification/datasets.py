import os
import json
import torch
import numpy as np
import scipy.sparse as sp
from config import args
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from collections import Counter


'''
three datasets:

DD : 
     node_labels (using embedding, as features)
     adjacency_matrix
     graph_indicator (labeling the nodes belongs to which graph)
     graph_labels (labels)
     
PROTEINS/ENZYMES:
     node_labels
     node_attributes (using as features)
     adjacency_matrix
     grahp_indicator
     graph_labels
'''

class get_data(object):
    def __init__(self, data_name = "DD", data_root = "data", val_size = 0.1, test_size = 0.1):
        self.data_root = data_root
        self.data_name = data_name
        adj, graph_indicator, graph_labels, node_labels, node_attributes = self.read_data()
        #self.adj = adj.tocsr()
        self.node_labels = node_labels
        if data_name != "DD":
            #self.features = normalized_feature(node_attributes)
            '''
            row_max = node_attributes.max(0)
            row_min = node_attributes.min(0)
            print(row_max)
            print(row_min)
            print((node_attributes - row_min)/row_max)
            node_attributes = (node_attributes - row_min)/row_max
            '''
            self.features = node_attributes

        self.graph_indicator =  graph_indicator

        self.graph_labels = graph_labels
        self.graph_len = len(graph_labels)
        self.train_index, self.val_index, self.test_index = self.split_data(val_size, test_size)

        self.train_label = graph_labels[self.train_index]
        self.val_label = graph_labels[self.val_index]
        self.test_label = graph_labels[self.test_index]

        #adj = adj + sp.eye(adj.shape[0])
        if args.model != 'gin':
            adj = normalized_adj(adj)

        self.adj = adj.tocsr()


    def split_data(self, val_size, test_size):
        unique_indicator = np.asarray(list(set(self.graph_indicator)))
        train, test = train_test_split(unique_indicator,
                                       test_size = test_size,
                                       random_state = 1234)
        train, val = train_test_split(train,
                                      test_size = val_size/(1. - test_size),
                                      random_state = 1234)
        return train, val, test

    def __getitem__(self, index):

        mask = self.graph_indicator == index
        node_labels = self.node_labels[mask]
        my_len = len(node_labels)

        #as attributions
        if self.data_name == 'DD':
            node_labels = self.node_labels[mask].astype(np.int64)
        else:
            node_labels = self.features[mask]

        graph_indicator = self.graph_indicator[mask]
        graph_labels = np.array([self.graph_labels[index]])
        sparse_adj = self.adj[mask, :][: , mask]

        return sparse_adj, node_labels, graph_indicator, graph_labels

    def __len__(self):
        return self.graph_len

    def read_data(self):

        str_name = self.data_name
        data_str = os.path.join(self.data_root, str_name)
        print("Loading {}_A.txt".format(str_name))

        adj = np.genfromtxt(os.path.join(data_str, "{}_A.txt".format(str_name)),
                            dtype = np.int64, delimiter = ',') - 1

        print("Loading {}_graph_indicator.txt".format(str_name))
        graph_indicator = np.genfromtxt(os.path.join(data_str,
                                                     "{}_graph_indicator.txt".format(str_name)),
                                        dtype = np.int64) - 1

        #print(Counter(graph_indicator))



        print("Loading {}_graph_labels.txt".format(str_name))
        graph_labels = np.genfromtxt(os.path.join(data_str,
                                                  "{}_graph_labels.txt".format(str_name)),
                                     dtype = np.int64) - 1
        #print(Counter(graph_labels))


        print("Loading {}_node_labels.txt".format(str_name))

        node_labels = np.genfromtxt(os.path.join(data_str,
                                                 "{}_node_labels.txt".format(str_name)),
                                    dtype = np.int64) - 1

        if str_name != "DD":
            print("Loading {}_node_attributes.txt".format(str_name))
            node_attributes = read_node_attributes(str_name)
        else:
            node_attributes = node_labels


        num_nodes = len(node_labels)
        sparse_adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])),
                                   shape = (num_nodes, num_nodes),
                                   dtype = np.float32)

        sparse_adj = sparse_adj \
                     + sparse_adj.T.multiply(sparse_adj.T > sparse_adj) \
                     - sparse_adj.multiply(sparse_adj.T > sparse_adj)
        print("num of edges", sparse_adj.getnnz())



        return sparse_adj, graph_indicator, graph_labels, node_labels, node_attributes


#combine the batch matrix into a big matrix
'''
    i.e.
    A1, A2, A3, ...AN
    (A1
       A2
         A3
           A4...
                AN)
'''
def combine_matrix(lst):

    size = 0
    size_lst = [0]
    for i in range(len(lst)-1):
        size += lst[i].shape[0]
        size_lst.append(size)

    size += lst[-1].shape[0]

    row, col = [x.row for x in lst], [x.col for x in lst]

    data = np.concatenate([x.data for x in lst], axis = 0).astype(np.float32)
    row = np.concatenate([(r + m) for (r, m) in zip(row, size_lst)], axis = 0)
    col = np.concatenate([(c + m) for (c, m) in zip(col, size_lst)], axis = 0)
    return sp.coo_matrix((data, (row, col)), (size, size))


def get_sptensor(input, device):
    idx = torch.LongTensor(np.vstack((input.row, input.col))).to(device)
    val = torch.from_numpy(input.data).to(device)
    out = torch.sparse.FloatTensor(idx, val, input.shape).to(device)
    return out

def normalized_adj(adj):
    '''n_adj = D^-0.5 * adj * D^-0.5'''
    adj = adj.tocoo()
    rowsums = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsums, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt).tocoo()

def normalized_feature(features):
    rowsums = np.array(abs(features).sum(1))
    d_inv = np.power(rowsums, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0
    d_inv = sp.diags(d_inv)
    return d_inv.dot(features)


def array2tensor(input, device):
    '''
    :param input:
            a tuple consisting of
            (sparse_adj, node_labels, graph_indicator, graph_labels)
    :return:
    '''
    adj, node_labels, graph_indicator, graph_labels, n_norm, index = input
    adj = get_sptensor(adj, device)
    #node_labels = get_sptensor(node_labels, device)
    if args.dataset == 'DD':
        node_labels = torch.LongTensor(node_labels).to(device)
    else:
        node_labels = torch.tensor(node_labels).to(device)
    graph_indicator = torch.tensor(graph_indicator).to(device)
    graph_labels = torch.tensor(graph_labels).to(device)
    n_norm = torch.tensor(n_norm).to(device)
    #index = torch.tensor(index).to(device)

    adj = Variable(adj)
    node_labels = Variable(node_labels)
    graph_indicator = Variable(graph_indicator)
    graph_labels = Variable(graph_labels)
    n_norm = Variable(n_norm)
    #index = Variable(index)

    return adj, node_labels, graph_indicator, graph_labels, n_norm, index

# to combine all subgraphs into one big graph and calculate the
# sparse_adj, node_labels, graph_indicator, graph_labels
def mydata_loader(batch_size, dataset, index, shuffle = True):
    '''
    :param dataset:
    :return: (sparse_adj, node_labels, graph_indicator, graph_label)
             in which
             sparse.tensor sparse_adj (a combined
             sparse.tensor node_labels,
             tensor graph_indicator,
             tensor graph_label
    '''
    device = torch.device('cpu')
    #train_idx = dataset.train_index
    load_index = index
    np.random.shuffle(load_index)

    train_size = len(load_index)
    for idx in range(0, train_size, batch_size):
        mx_idx = min(idx + batch_size, train_size)
        epoch_idx = load_index[idx: mx_idx]

        batch_graph = [dataset[i] for i in epoch_idx]
        #generate batch datas for batch graphs

        #combined matrix
        batch_matrix = [data[0].tocoo() for data in batch_graph]
        graph_size = [data[0].shape[0] for data in batch_graph]

        batch_matrix = combine_matrix(batch_matrix).astype(np.float32)

        #combined node_attributions
        batch_node_attr = np.concatenate([graph[1] for graph in batch_graph], axis = 0).astype(np.float32)

        #combined indicator
        batch_indicator = np.concatenate([graph[2] for graph in batch_graph], axis = 0)

        #combined labels
        batch_labels = np.concatenate([graph[3] for graph in batch_graph], axis = 0)

        #combined graph_norm
        batch_snorm = np.concatenate([np.full((size, 1), 1.0/size) for size in graph_size])
        batch_snorm = np.power(batch_snorm, 0.5).astype(np.float32)

        yield array2tensor((batch_matrix,
                            batch_node_attr,
                            batch_indicator,
                            batch_labels,
                            batch_snorm,
                            epoch_idx), device)


def read_node_attributes(data_name):

    val = []
    path = os.path.join("data", data_name)
    path = os.path.join(path, "{}_node_attributes.txt".format(data_name))
    with open(path, "r") as f:
        for line in f:
            line = line.strip('\n').split(',')
            line = map(lambda x: x.replace(' ', ''), line)
            val.append(list(map(float, line)))

    return np.array(val)




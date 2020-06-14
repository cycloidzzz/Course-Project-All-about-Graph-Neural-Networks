import matplotlib.pyplot as plt
import numpy as np

def parse_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(float(line.strip('\n')))
    return data

d_lst = ['cora', 'pubmed', 'citeseer']


for i in range(3):
    dataset = d_lst[i]
    plt.subplot(2, 3, i + 1)
    plt.title(dataset)
    if i == 0:
        plt.ylabel("train loss")
    plt.xlabel("epoch")
    data1 = parse_data("gat_{}_train_loss.txt".format(dataset))
    data2 = parse_data("gcn_{}_train_loss.txt".format(dataset))
    data3 = parse_data("gin_{}_train_loss.txt".format(dataset))
    plt.plot(np.array(range(len(data1))), np.array(data1), linewidth = 1.0, color = "red", label = "GAT")
    plt.plot(np.array(range(len(data2))), np.array(data2), linewidth = 1.0, color = "blue", label = "GCN")
    plt.plot(np.array(range(len(data3))), np.array(data3), linewidth=1.0, color="green", label="GIN")
    plt.legend(loc = "upper right")

    plt.subplot(2, 3, i + 4)
    if i == 0:
        plt.ylabel("valid accuracy")
    plt.xlabel("epoch")
    data1 = parse_data("gat_{}_valid_acc.txt".format(dataset))
    data2 = parse_data("gcn_{}_valid_acc.txt".format(dataset))
    data3 = parse_data("gin_{}_valid_acc.txt".format(dataset))
    if i == 0:
        plt.plot(np.array(range(len(data1))), np.array(data1), linewidth = 1.0, color = "red", label = "GAT")
        plt.plot(np.array(range(len(data2))), np.array(data2), linewidth = 1.0, color = "blue", label = "GCN")
        plt.plot(np.array(range(len(data3))), np.array(data3), linewidth=1.0, color="green", label="GIN")
    else:
        plt.plot(np.array(range(len(data2))), np.array(data2), linewidth=1.0, color="blue", label="GCN")
        plt.plot(np.array(range(len(data1))), np.array(data1), linewidth=1.0, color="red", label="GAT")
        plt.plot(np.array(range(len(data3))), np.array(data3), linewidth=1.0, color="green", label="GIN")
    plt.legend(loc = "lower right")

'''
dataset = d_lst[2]
plt.subplot(2, 1, 1)
plt.title(dataset)
plt.ylabel("train loss")
plt.xlabel("epoch")
data1 = parse_data("gat_{}_train_loss.txt".format(dataset))
data2 = parse_data("gcn_{}_train_loss.txt".format(dataset))
plt.plot(np.array(range(len(data1))), np.array(data1), linewidth=1.0, color="red", label="GAT")
plt.plot(np.array(range(len(data2))), np.array(data2), linewidth=1.0, color="blue", label="GCN")
plt.legend(loc="upper right")

plt.subplot(2, 1, 2)
plt.ylabel("valid accuracy")
plt.xlabel("epoch")
data1 = parse_data("gat_{}_valid_acc.txt".format(dataset))
data2 = parse_data("gcn_{}_valid_acc.txt".format(dataset))
plt.plot(np.array(range(len(data1))), np.array(data1), linewidth=1.0, color="red", label="GAT")
plt.plot(np.array(range(len(data2))), np.array(data2), linewidth=1.0, color="blue", label="GCN")
plt.legend(loc="lower right")
'''
plt.show()

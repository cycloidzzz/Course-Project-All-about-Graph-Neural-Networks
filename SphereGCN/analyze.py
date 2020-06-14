import numpy as np
import matplotlib.pyplot as plt


def parse_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(float(line.strip('\n')))
    return data


layer_lst = [20, 40, 10, 30]
color_lst1 = ['b--', 'c--', 'g--', 'r--']
color_lst2 = ['b-', 'c-', 'g-', 'r-']
color_lst3 = ['b:', 'c:', 'g:', 'r:']


def paint_loss():

    plt.ylabel("train loss")
    plt.xlabel("epoch")
    plt.ylim(0, 2.0)
    layer_lst = [40, 30, 20, 10]
    for i in range(4):
        layers = layer_lst[i]
        model_data = "res_{}_gcn_{}_train_loss.txt".format(str(layers), dataset)
        model_name = "{} layer res gcn".format(str(layers))
        result = np.array(parse_data(model_data))
        result = result if len(result) < 400 else result[0:399]
        plt.plot(result,  color_lst2[i], label = model_name)

    layer_lst = [40, 30, 20, 10]
    for i in range(4):
        layers = layer_lst[i]
        model_data = "sp_norm_{}_gcn_{}_train_loss.txt".format(str(layers), dataset)
        model_name = "{} layer normed gcn".format(str(layers))
        result = np.array(parse_data(model_data))
        result = result if len(result) < 400 else result[0:399]
        plt.plot(result, color_lst1[i], label = model_name)


def paint_acc():

    plt.ylabel("validation accuracy")
    plt.xlabel("epoch")
    layer_lst = [30, 40, 20, 10]
    for i in range(4):
        layers = layer_lst[i]
        model_data = "sp_norm_{}_gcn_{}_valid_acc.txt".format(str(layers), dataset)
        model_name = "{} layer normed gcn".format(str(layers))
        result = np.array(parse_data(model_data))
        result = result if len(result) < 400 else result[0:399]
        plt.plot(result, color_lst2[i], label=model_name)

    layer_lst = [10, 20, 30, 40]
    for i in range(4):
        layers = layer_lst[i]
        model_data = "res_{}_gcn_{}_valid_acc.txt".format(str(layers), dataset)
        model_name = "{} layer res gcn".format(str(layers))
        result = np.array(parse_data(model_data))
        result = result if len(result) < 400 else result[0:399]
        plt.plot(result, color_lst1[i], label=model_name)


dataset = "citeseer"

plt.title(dataset)

#paint_loss()

def test_acc():
    plt.xlabel("layers")
    plt.ylabel("test accuracy")
    set0 = [10, 20, 30, 40]
    '''
    set1 = [0.63199, 0.64000, 0.62799, 0.63899]
    set2 = [0.585000, 0.595000, 0.551999, 0.598000]
    set3 = [0.760000, 0.406999, 0.406999, 0.406999]
    '''
    set1 = [0.60900, 0.62499, 0.62399, 0.61900]
    set2 = [0.5340000, 0.5360000, 0.5140000, 0.4790000]
    set3 = [0.4900, 0.1600, 0.1600, 0.1600]
    plt.plot(set0, set1, 'r-x', label = 'normed gcn')
    plt.plot(set0, set2, 'g-o', label = 'bn_res gcn')
    plt.plot(set0, set3, 'b-^', label = 'res gcn')


test_acc()
plt.legend(loc = "lower right")
plt.show()

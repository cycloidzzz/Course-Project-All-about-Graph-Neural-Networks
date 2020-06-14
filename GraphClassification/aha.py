import numpy as np
import os

def read_node_attributes():
    '''

    :return: the node attributes as np.array
    '''
    val = []
    path = os.path.join("data\\PROTEINS", "PROTEINS_node_attributes.txt")
    with open("data/PROTEINS/PROTEINS_node_attributes.txt", "r") as f:
        for line in f:
            line = line.strip('\n').split(',')
            line = map(lambda x: x.replace(' ', ''), line)
            val.append(list(map(float, line)))
    return np.array(val)

val = read_node_attributes()

print(val.max(0))
print(len(val/val.max(0)))
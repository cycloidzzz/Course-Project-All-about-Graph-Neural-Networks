#dealt with problems in  gat_PROTEINS_VAL_ACC and gcn_DD_val_acc

def parse_file(filename):
    data_lst = []
    with open(filename , "r") as f:
        for line in f:
            data_lst.append(float(line.strip('\n')[7:-1]))

    return data_lst

def write_file(filename):
    data_lst = parse_file(filename)
    with open(filename, "w") as f:
        for num in data_lst:
            f.write(str(num) + '\n')



import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default = 'DD')
args.add_argument('--model', default = 'gcn')
args.add_argument('--method', default = 'mini-batch')
args.add_argument('--norm_type', default = 'b_norm')
args.add_argument('--learning_rate', type = float, default = 7e-4) #1e-3
args.add_argument('--epoches', type = int, default = 800)
args.add_argument('--layers', type = int, default = 4)
args.add_argument('--batch_size', type = int, default = 20)
#128 for GCN/GIN, 16 for GAT
args.add_argument('--hidden', type = int, default = 16) #40 for GCN
args.add_argument('--dropout', type = float, default = 0.0)
args.add_argument('--weight_decay', type = float, default = 0.0)
args.add_argument('--early_stopping', type = int, default = 100)
args.add_argument('--max_degree', type = int, default = 3)

args = args.parse_args()
print(args)


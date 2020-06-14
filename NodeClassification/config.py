import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default = 'cora')
args.add_argument('--model', default = 'gcn')
args.add_argument('--method', default = 'whole-batch')
args.add_argument('--learning_rate', type = float, default = 0.01)
args.add_argument('--epoches', type = int, default = 800)
args.add_argument('--batch_size', type = int, default = 100)
args.add_argument('--hidden', type = int, default = 24) #8 for GAT
args.add_argument('--dropout', type = float, default = 0.5)
args.add_argument('--weight_decay', type = float, default = 5e-4)
args.add_argument('--patience', type = int, default = 100)
args.add_argument('--alpha', type = float, default = 0.2)
args.add_argument('--nheads', type = int, default = 8)
args.add_argument('--max_degree', type = int, default = 3)

args = args.parse_args()
print(args)


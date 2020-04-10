import argparse
import time 
import numpy as np 
import os
import torch 
from pprint import pprint 
from dataloader import DataLoaderPolyvore
from model import CompatibilityGAE
from utils.misc import compute_degree_support, normalize_nonsym_adj, support_dropout, csr_to_sparse_tensor
from copy import deepcopy

# set random seed
# seed = int(time.time())  # 12342
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='visual compatibility')
parser.add_argument('-d', '--dataset', type=str, default='polyvore', choices=['polyvore', 'fashiongen'])
parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='learning rate')
parser.add_argument('-wd', '--weight-decay', type=float, default=0., help='weight decay')
parser.add_argument('-e', '--epochs', type=int, default=4000, help='Number of training epochs')
parser.add_argument('-hi', '--hidden', type=int, nargs='+', default=[350, 350, 350], help='Number of hidden units in the GCN layers.')
parser.add_argument('-drop', '--dropout', type=float, default=0.5, help='dropout probability')
parser.add_argument('-deg', '--degree', type=int, default=1, help='degree of convolution, i.e., number of support nodes.')
parser.add_argument('-sd', '--support-dropout', type=float, default=0.15, help='Use dropout on support adjacency matrix, dropping all the connections from some nodes')
parser.add_argument('--data-dir', type=str, default='/home/alan/Downloads/fashion/polyvore/dataset2')
parser.add_argument('--save-path', type=str, default='/home/alan/Downloads/fashion/polyvore/')
parser.add_argument('--device', type=str, default='cuda:0')

mg = parser.add_mutually_exclusive_group(required=False)
mg.add_argument('-bn', '--batch-norm', action='store_true', help='Option to turn on batchnorm in GCN layers')
mg.add_argument('-no-bn', '--no-batch-norm', action='store_false', help='Option to turn off batchnorm in GCN layers')
parser.set_defaults(batch_norm=True)

args = parser.parse_args()
args = vars(args)

print("\nSetting: \n")
for key, val in args.items():
    print("{}: {}".format(key, val))
print()

# Define parameters
DATASET = args['dataset']
NB_EPOCHS = args['epochs']
DROP = args['dropout']
HIDDEN = args['hidden']
LR = args['learning_rate']
NUM_CLASSES = 2  # each edge is either connected or not 
DEGREE = args['degree']
BATCH_NORM = args['batch_norm']
SUP_DROP = args['support_dropout'] 
ADJ_SELF_CONNECTIONS = True 
DATA_DIR = args['data_dir']
WD = args['weight_decay']
DEVICE = args['device']

# prepare dataset
if DATASET in ('polyvore', 'fashiongen'):
    if DATASET == 'polyvore':
        dl = DataLoaderPolyvore(DATA_DIR)
    else:
        raise NotImplementedError('Support to fashiongen dataset will be added soon!')

    # node features, message-passing adj, ground-truth labels, start node idx, end node idx of edges to evaluate loss
    #  > message-passing(?메세지전달) adj에서 message-passing의 의미는?    
    train_features, train_mp_adj, train_labels, train_row_idx, train_col_idx = dl.get_phase('train')
    # train_mp_adj 는 나머지 half graph 정보 = message-passing adj
    # print(train_mp_adj.shape) # (84497, 84497)
    # train_labels, train_row_idx, train_col_idx
    #   - graph의 node들의 개수가 338488 인데 이들 반절은 pos vs neg로 만든 형태
    # print(train_labels.shape) # (338488,)
    # print(train_row_idx.shape) # (338488,)
    # print(train_row_idx) # [24749. 39950. 36779. ...  5722. 60576. 44047.]
    # print(train_col_idx.shape) # (338488,)
    # print(train_col_idx) # [12189. 39947.  8606. ...  5511.  8237. 20453.]

    val_features, val_mp_adj, val_labels, val_row_idx, val_col_idx = dl.get_phase('valid')

    # normalize features
    train_features, mean, std = dl.normalize_feature(train_features, return_moments=True)
    # print(train_features.shape) # (84497, 2048)
    # mean, std > predict에서 필요할것으로 보여 실제로는 저장되어야할 정보
    # print('train feature : ' ,  mean, std) # [[0.59618651 0.64498316 0.68714354 ... 0.32978797 0.72110322 0.57665217]] [[0.51395941 0.53921698 0.53976958 ... 0.2645195  0.86538214 0.44306204]]
    # print('\t' , mean.shape, std.shape) # (1, 2048) (1, 2048)
    
    val_features              = dl.normalize_feature(val_features, mean=mean, std=std, return_moments=False)

else:
    raise NotImplementedError('Dataloader for dataset {} is not supported yet!'.format(DATASET))

# convert features to tensors
train_features = torch.from_numpy(train_features).to(DEVICE)
val_features   = torch.from_numpy(val_features).to(DEVICE)
train_labels   = torch.from_numpy(train_labels).float().to(DEVICE)
val_labels     = torch.from_numpy(val_labels).float().to(DEVICE)

train_row_idx  = torch.from_numpy(train_row_idx).long().to(DEVICE)
train_col_idx  = torch.from_numpy(train_col_idx).long().to(DEVICE)
val_row_idx    = torch.from_numpy(val_row_idx).long().to(DEVICE)
val_col_idx    = torch.from_numpy(val_col_idx).long().to(DEVICE)

# get support adjacency matrix [A0, ..., AS] 
train_support = compute_degree_support(train_mp_adj, DEGREE, adj_self_connections=ADJ_SELF_CONNECTIONS)
# print(len(train_support)) # 2
# print(train_support[0].todense()) # 0 위치에 I 행렬을 가지고 있다.
# (84497, 84497)
# [[1. 0. 0. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 0. ... 0. 1. 0.]
#  [0. 0. 0. ... 0. 0. 1.]]
# print(train_support[1].todense()) # 0 실제 graph metrix 정보
# (84497, 84497)
# [[1. 0. 1. ... 0. 0. 0.]
#  [0. 1. 1. ... 0. 0. 0.]
#  [1. 1. 1. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 1. 1. 1.]
#  [0. 0. 0. ... 1. 1. 0.]
#  [0. 0. 0. ... 1. 0. 1.]]

val_support   = compute_degree_support(val_mp_adj, DEGREE, adj_self_connections=ADJ_SELF_CONNECTIONS)
# val_support[0] == val_support[1] == (8923, 8923)

# normalize these support adjacency matrices, except the first one which is symmetric
for i in range(1, len(train_support)):
    # print(i, train_support[i].shape) # 1 (84497, 84497)
    # print(train_support[i].toarray())    
    # [[1. 0. 1. ... 0. 0. 0.]
    #  [0. 1. 1. ... 0. 0. 0.]
    #  [1. 1. 1. ... 0. 0. 0.]
    #  ...
    #  [0. 0. 0. ... 1. 1. 1.]
    #  [0. 0. 0. ... 1. 1. 0.]
    #  [0. 0. 0. ... 1. 0. 1.]]
    train_support[i] = normalize_nonsym_adj(train_support[i])
    # print(i, train_support[i].shape) # 1 (84497, 84497) 
    # print(train_support[i].toarray()) # normalize 되어 나온다.
    # [[0.2        0.         0.2        ... 0.         0.         0.        ]
    #  [0.         0.01176471 0.01176471 ... 0.         0.         0.        ]
    #  [0.16666667 0.16666667 0.16666667 ... 0.         0.         0.        ]
    #  ...
    #  [0.         0.         0.         ... 0.16666667 0.16666667 0.16666667]
    #  [0.         0.         0.         ... 0.2        0.2        0.        ]
    #  [0.         0.         0.         ... 0.16666667 0.         0.16666667]]

    val_support[i]   = normalize_nonsym_adj(val_support[i])  

val_support = [csr_to_sparse_tensor(adj).to(DEVICE) for adj in val_support]

num_supports = len(train_support)

settings = {
    'num_support'  : num_supports,  
    'dropout'      : DROP,
    'batch_norm'   : BATCH_NORM,
    'learning_rate': LR,
    'wd'           : WD,
}


# create model
model = CompatibilityGAE(
    input_dim   = train_features.shape[1],
    hidden      = HIDDEN,
    num_classes = 2,
    settings    = settings
)
model.to(DEVICE)
model.train()

print("\nModel: ")
print(model)

best_val_acc       = 0
best_epoch         = 0
best_val_train_acc = 0
best_train_acc     = 0 

tmp = csr_to_sparse_tensor(train_support[0]).to(DEVICE)
for epoch in range(NB_EPOCHS):
    if SUP_DROP > 0:
        # do not modify the first support adj matrix, which is self-connections
        epoch_train_supports = [tmp] # do not miss the first self-connection edges
        for i in range(1, len(train_support)):
            sampled_adj = support_dropout(train_support[i], SUP_DROP, drop_edge=True)
            # binarilize the support adjacency matrix
            sampled_adj.data[...] = 1 
            sampled_adj = normalize_nonsym_adj(sampled_adj)
            # convert adj sparse matrix to sparse tensor
            sampled_adj = csr_to_sparse_tensor(sampled_adj).to(DEVICE)
            # update message passing graph feed to the model
            epoch_train_supports.append(sampled_adj)
    else:
        epoch_train_supports = [csr_to_sparse_tensor(adj).to(DEVICE) for adj in train_support]
    # run one epoch
    train_avg_loss, train_acc = model.train_epoch(train_features, epoch_train_supports, train_row_idx, train_col_idx, train_labels)
    with torch.no_grad():
        model.eval()
        valid_pred = model.predict(val_features, val_support, val_row_idx, val_col_idx)
        valid_acc  = model.accuracy(valid_pred, val_labels)
    model.train()

    print("Epoch: {} -- Train Loss: {:.4f} -- Train Acc: {:.4f} -- Valid Acc: {:.4f}".format(epoch, train_avg_loss, train_acc, valid_acc))

    if valid_acc > best_val_acc:
        best_val_acc       = valid_acc
        best_epoch         = epoch 
        best_val_train_acc = train_acc 
        # save model
        torch.save(
            {
                'state_dict': model.state_dict(),
                'args'      : args
            }, 
            os.path.join(args['save_path'], 'best_model.pth')
        )
    
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    
print("Training Done!")
print("Best Epoch: {} -- Best Valid Acc: {:.4f} -- Best Train Acc at epoch {}: {:.4f} -- Best Overall Train Acc: {:.4f}".format(
    best_epoch, best_val_acc, best_epoch, best_val_train_acc, best_train_acc))

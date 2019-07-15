import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import utils.gsn_argparse as gap

import numpy as np
from torch_geometric.data import Data

import os
import ssl
import random

import trainer

ssl._create_default_https_context = ssl._create_unverified_context

DATASET = "imdb"


# torch.manual_seed(31415926)


def _load_split_data():
    res = []
    for filepath in ["./data/aclImdb/train",
                     "./data/aclImdb/test"]:
        bert_encode_res = np.load(os.path.join(filepath, "split_bert_large_encode_res.npy"),
                                  allow_pickle=True)  # 25000,768
        y = np.load(os.path.join(filepath, "split_y.npy"), allow_pickle=True)  # 25000
        edge = np.load(os.path.join(filepath, "split_edge.npy"), allow_pickle=True)  # 2,num_edge*2
        datas = []
        for x, y, e in zip(bert_encode_res, y, edge):
            x = [_.tolist() for _ in x]
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            if len(e) == 0:
                e = torch.empty((0, 2), dtype=torch.long).t()
            else:
                e = torch.tensor(e, dtype=torch.long).t()
            datas.append(Data(x=x, edge_index=e, y=y))
        res.append(datas)
    return res[0], res[1]


def _load_split_2k_data():
    res = []
    for filepath in ["./data/aclImdb/train",
                     "./data/aclImdb/test"]:
        bert_encode_res = np.load(os.path.join(filepath, "split_2k_bert_large_encode_res.npy"),
                                  allow_pickle=True)  # 25000,768
        y = np.load(os.path.join(filepath, "split_2k_y.npy"), allow_pickle=True)  # 25000
        edge = np.load(os.path.join(filepath, "split_2k_edge.npy"), allow_pickle=True)  # 2,num_edge*2
        datas = []
        for x, y, e in zip(bert_encode_res, y, edge):
            x = [_.tolist() for _ in x]
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            if len(e) == 0:
                e = torch.empty((0, 2), dtype=torch.long).t()
            else:
                e = torch.tensor(e, dtype=torch.long).t()
            datas.append(Data(x=x, edge_index=e, y=y))
        res.append(datas)
    return res[0], res[1]


def _load_split_star_data():
    res = []
    for filepath in ["./data/aclImdb/train",
                     "./data/aclImdb/test"]:
        bert_encode_res = np.load(os.path.join(filepath, "split_bert_encode_res.npy"))  # 25000,768
        y = np.load(os.path.join(filepath, "split_y.npy"))  # 25000
        edge = np.load(os.path.join(filepath, "split_edge.npy"))  # 2,num_edge*2
        datas = []
        for x, y, e in zip(bert_encode_res, y, edge):
            x = [_.tolist() for _ in x]
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            star = x.mean(dim=0).view(1, -1)
            x = torch.cat([x, star], dim=0)
            y = torch.cat([y, torch.tensor([-1], dtype=torch.long)])
            a = torch.full((1, len(x) - 1), len(x) - 1).long()
            b = torch.tensor(range(len(x) - 1)).view(1, -1).long()
            e = torch.cat([torch.cat([a, b], dim=0), torch.cat([b, a], dim=0)], dim=1)
            datas.append(Data(x=x, edge_index=e, y=y))
        res.append(datas)
    return res[0], res[1]


def _load_all_data():
    res = []
    for filepath in ["./data/aclImdb/train",
                     "./data/aclImdb/test"]:
        bert_encode_res = np.load(os.path.join(filepath, "all_bert_large_encode_res.npy"))  # 25000,768
        y = np.load(os.path.join(filepath, "all_y.npy"))  # 25000
        edge = np.load(os.path.join(filepath, "all_edge.npy"))  # 2,num_edge*2
        # if filepath.endswith("test"):
        #     datas = [Data(x=torch.from_numpy(bert_encode_res), edge_index=torch.empty((2,0)).long(),
        #                   y=torch.from_numpy(y).long())]
        # else:
        datas = [Data(x=torch.from_numpy(bert_encode_res), edge_index=torch.from_numpy(edge),
                      y=torch.from_numpy(y).long())]
        res.append(datas)
    return res[0], res[1]


def load_data(data_type):
    bs = 0
    shuffle = True
    if data_type == "split":
        trainData, testData = _load_split_data()
        bs = 5000
    elif data_type == "split_2k":
        trainData, testData = _load_split_2k_data()
        bs = 100
        valData = trainData[0:1]
        trainData = trainData[1: ]
        bs = 1
        shuffle = False
    elif data_type == "split_star":
        trainData, testData = _load_split_star_data()
        bs = 80
    elif data_type == "all":
        trainData, testData = _load_all_data()
        bs = 1
    else:
        trainData, testData = _load_all_data()
        bs = 1
    train_loader = DataLoader(trainData, batch_size=bs, shuffle=shuffle)
    val_loader = DataLoader(valData, batch_size=bs, shuffle=shuffle)
    test_loader = DataLoader(testData, batch_size=bs, shuffle=shuffle)

    return train_loader, val_loader, test_loader


def main(_args):
    args = gap.parser.parse_args(_args)

    args.device = 4
    args.dropout = 0.2
    args.hidden = 1024
    args.num_layers = 3
    args.graph_type = "split_2k"
    args.cross_layer = False
    args.lr = 0.0001
    args.layer_norm_star = False
    args.layer_norm = False
    args.additional_self_loop_relation_type = True
    args.additional_node_to_star_relation_type = True

    train_loader, val_loader, test_loader = load_data(args.graph_type)

    trainer.trainer(args, DATASET, train_loader, val_loader, test_loader,
                    num_features=1024,
                    num_node_class=2,
                    max_epoch=args.epochs,
                    node_multi_label=False)


if __name__ == '__main__':
    main(sys.argv[1:])

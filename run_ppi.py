import sys
import torch
from torch_geometric.data import DataLoader
from module.graph_star import GraphStar
import utils.tensorboard_writer as tw

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import PPI
from utils.gsn_argparse import str2bool, str2actication
import os.path as osp
from sklearn import metrics
import trainer

import time
import ssl
import utils.gsn_argparse as gap

ssl._create_default_https_context = ssl._create_unverified_context

DATASET = "ppi"


def load_data():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'PPI')

    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='test')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def main(_args):
    args = gap.parser.parse_args(_args)

    args.device = 1
    args.dropout = 0.2
    args.hidden = 512
    args.num_layers = 3
    args.cross_layer = False
    args.additional_self_loop_relation_type = True
    args.additional_node_to_star_relation_type = True

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_data()

    trainer.trainer(args, DATASET, train_loader, val_loader, test_loader,
                    num_features=train_dataset.num_features, max_epoch=args.epochs,
                    num_node_class=train_dataset.num_classes,
                    node_multi_label=True)


if __name__ == '__main__':
    main(sys.argv[1:])

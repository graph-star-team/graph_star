import sys
import torch
from torch_geometric.data import DataLoader
import utils.gsn_argparse as gap

from torch_geometric.datasets import Planetoid
import os.path as osp
import torch_geometric.transforms as T

import ssl
import trainer

ssl._create_default_https_context = ssl._create_unverified_context

def load_data(dataset_name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)

    dataset = Planetoid(path, dataset_name, T.TargetIndegree())
    train_loader = DataLoader(dataset, batch_size=1)
    return dataset, train_loader


def main(_args):
    args = gap.parser.parse_args(_args)

    dataset, train_loader = load_data(args.dataset)
    trainer.trainer(args, args.dataset, train_loader, train_loader, train_loader, transductive=True,
                    num_features=dataset.num_features,
                    num_node_class=dataset.num_classes,
                    max_epoch=args.epochs,
                    node_multi_label=False)


if __name__ == '__main__':
    main(sys.argv[1:])

import sys
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import os.path as osp

import trainer

import ssl
import utils.gsn_argparse as gap


ssl._create_default_https_context = ssl._create_unverified_context


# better to use *_no_repeat file
DATASET = "R8" # should be in ['R8', 'R52', 'mr', 'ohsumed', '20ng']

def load_data():
    bs = 96
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DATASET)
    dataset = TUDataset(path, DATASET, use_node_attr=True)
#     dataset = dataset.shuffle()
#     dataset.data.x = dataset.data.x[:, :-3]
    # dataset.data.node_label = dataset.data.x[:, -3:]
    
    if 'MR' in DATASET:
        real_train_size = 6398
        train_size = 7108
    elif 'R8' in DATASET:
        real_train_size = 4937
        train_size = 5485
    elif '20ng' in DATASET:
        real_train_size = 10183
        train_size = 11314
    elif 'R52' in DATASET:
        real_train_size = 5879
        train_size = 6532
    elif 'ohsumed' in DATASET:
        real_train_size = 3022
        train_size = 3357
    
    train_loader = DataLoader(dataset[:real_train_size], batch_size=bs)
    val_loader = DataLoader(dataset[real_train_size:train_size], batch_size=bs)
    test_loader = DataLoader(dataset[train_size:], batch_size=bs)

    print('batch size is : ' + str(bs))
    return dataset, dataset, dataset, train_loader, val_loader, test_loader


def main(_args):
    args = gap.parser.parse_args(_args)

    args.device = 3
    args.dropout = 0.3
    args.coef_dropout = 0.3
    args.hidden = 512
    args.num_layers = 3
    args.l2 = 0.002
    args.layer_norm_star = True # For MR this should be false
    args.layer_norm = True # For MR this should be false
    args.cross_layer = False
    args.cross_star = False
    args.lr = 0.0001
    args.additional_self_loop_relation_type = True
    args.additional_node_to_star_relation_type = True

    # ohsumed
#     args.dropout = 0.3
#     args.hidden = 512
#     args.num_layers = 3
#     args.l2 = 0.0002
#     args.layer_norm_star = True
#     args.layer_norm = True
#     args.cross_layer = False
#     args.cross_star = False
#     args.lr = 0.0001
#     args.additional_self_loop_relation_type = True
#     args.additional_node_to_star_relation_type = True

    print('gpu device is: ' + str(args.device))
    print('num of star is: ' + str(args.num_star))
    print('dropout is: ' + str(args.dropout))
    print('num of heads is ' + str(args.heads))
    print('num of hidden unit is ' + str(args.hidden))
    print('l2 is : ' +str(args.l2))
    print('num of relations: ' + str(args.num_relations))
    print('cross star: ' + str(args.cross_star))
    print('normal node layer norm: ' + str(args.layer_norm))
    print('star layer norm: ' + str(args.layer_norm_star))
    print('learning rate is: ' + str(args.lr))
    print('whether to use init embdz: ' + str(args.use_e))
    print('num of layers: ' + str(args.num_layers))
    print('whether use cross layer: ' + str(args.cross_layer))
    print('whether to add self loop weight: ' + str(args.additional_self_loop_relation_type))
    print('whether to add star-node weight: ' + str(args.additional_node_to_star_relation_type))
    print('star init methods is: ' + args.star_init_method) # ["mean", "attn"]
    print('relation_score_function is: ' + args.relation_score_function) # ComplEx, RotatE, pRotatE

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_data()

    trainer.trainer(args, DATASET, train_loader, val_loader, test_loader,
                    num_features=train_dataset.num_features,
                    num_graph_class=train_dataset.num_classes,
                    node_multi_label=False,
                    max_epoch=args.epochs,
                    graph_multi_label=False)


if __name__ == '__main__':
    main(sys.argv[1:])

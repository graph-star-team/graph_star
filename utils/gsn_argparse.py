import torch.nn.functional as F
import argparse
from texttable import Texttable
from torch import device

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def str2actication(v):
    if v.lower() == "relu":
        return F.relu
    if v.lower() == "elu":
        return F.elu
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), str(args[k])] for k in keys])
    print(t.draw())


def tempDevice(x):
    """
    Function to faciliate using both a GPU (defined as an int)
    and a CPU (defined as a torch.device object)
    :param x: 
    """
    if type(x) == int:
       return x
    elif x == 'cpu':
       return device('cpu')
    return 0  #default value


parser = argparse.ArgumentParser(description='GSN args.')
parser.add_argument('--device', type=tempDevice, default="0")
parser.add_argument('--num_star', type=int, default=1)
parser.add_argument('--num_relations', type=int, default=1)
parser.add_argument('--one_hot_node', type=str2bool, default=False)
parser.add_argument('--one_hot_node_num', type=int, default=0)
parser.add_argument('--cross_star', type=str2bool, default=True)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--coef_dropout', type=float, default=0)
parser.add_argument('--residual', type=str2bool, default=True)
parser.add_argument('--residual_star', type=str2bool, default=True)
parser.add_argument('--layer_norm', type=str2bool, default=True)
parser.add_argument('--layer_norm_star', type=str2bool, default=True)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--use_e', type=str2bool, default=False)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--hidden', type=int, default=1024)
parser.add_argument('--activation', type=str2actication, default="elu")
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--cross_layer', type=str2bool, default=True)
parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--additional_self_loop_relation_type', type=str2bool, default=True)
parser.add_argument('--additional_node_to_star_relation_type', type=str2bool, default=True)
parser.add_argument('--star_init_method', type=str, default="attn")
parser.add_argument('--relation_score_function', type=str, default="DistMult",
                    help="DistMult")
parser.add_argument('--dataset', type=str,default="")
parser.add_argument('--epochs', type=int, default=2000)

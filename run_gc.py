import sys
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import os.path as osp

import trainer
import time

import ssl
import utils.gsn_argparse as gap
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context


def load_data(dataset_name, val_idx):
    bs = 600
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)
    dataset = TUDataset(path, dataset_name, use_node_attr=True)
    dataset = dataset.shuffle()

    if dataset_name == "DD" or dataset_name == "MUTAG":
        # remove node label
        dataset.data.x = dataset.data.x[:, :-3]

    val_size = len(dataset) // 10

    if val_idx == 0:
        train_data = dataset[val_size:]
    elif 0 < val_idx < 9:
        train_data = dataset[:(val_idx * val_size)] + dataset[((val_idx + 1) * val_size):]
    elif val_idx == 9:
        train_data = dataset[:(val_idx * val_size)]
    else:
        raise AttributeError("val index must in [0,9]")

    train_loader = DataLoader(train_data, batch_size=bs)
    val_loader = DataLoader(dataset[(val_idx * val_size): ((val_idx + 1) * val_size)], batch_size=bs)
    test_loader = DataLoader(dataset[(val_idx * val_size): ((val_idx + 1) * val_size)], batch_size=bs)

    return dataset, train_loader, val_loader, test_loader


def main(_args):
    args = gap.parser.parse_args(_args)

    val_accs = []
    all_gc_accs = []

    for i in range(10):
        start_time = time.perf_counter()
        dataset, train_loader, val_loader, test_loader = load_data(args.dataset, i)

        val_acc, gc_accs = trainer.trainer(args, args.dataset, train_loader, val_loader, test_loader,
                                           num_features=dataset.num_features,
                                           num_graph_class=dataset.num_classes,
                                           max_epoch=args.epochs,
                                           node_multi_label=False,
                                           graph_multi_label=False)
        val_accs.append(val_acc)
        all_gc_accs.append(np.array(gc_accs))
        end_time = time.perf_counter()
        spent_time = (end_time - start_time) / 60
        print(" It took: {:2f} minutes to complete one round....".format(spent_time))
        print("\033[1;32m Best graph classification accuracy in {}th round is: {:4f} \033[0m".format((i + 1), val_acc))

    all_gc_accs = np.vstack(all_gc_accs)
    all_gc_accs = np.mean(all_gc_accs, axis=0)
    final_gc = np.mean(val_accs)
    print("\n\n\033[1;32m Average over 10 best results:  {:.4f}  \033[0m".format(final_gc))
    val_accs = ['{:.4f}'.format(i) for i in val_accs]
    print(" 10 Best results: ", np.asfarray(val_accs, float))
    print(" DiffPoll cross val:  {:.4f} ".format(np.max(all_gc_accs)))
    print(" DiffPoll argmax pos: ", np.argmax(all_gc_accs))


if __name__ == '__main__':
    main(sys.argv[1:])

import sys

from torch_geometric.data import Data
from utils.gsn_argparse import str2bool, str2actication
import torch_geometric.utils as gutils
import os.path as osp

import ssl
import torch
from torch_geometric.nn import GAE
import trainer
import utils.gsn_argparse as gap
import utils.label_encode_dataset as led
import utils.create_node_embedding as cne
import numpy as np
import pandas as pd
from sklearn import preprocessing
from gensim.models import KeyedVectors
import math as m



ssl._create_default_https_context = ssl._create_unverified_context





def load_data():
    entity_id = pd.read_csv('./data/FB15k/entities.txt', sep='\t', header=None, names=['entity', 'id'], engine='python')
    entity = entity_id['entity'].values

    relation_id = pd.read_csv('./data/FB15k/relations.txt', sep='\t', header=None, names=['relation', 'id'], engine='python')
    relation = relation_id['relation'].values

    data = pd.read_csv('./data/FB15k/valid.txt', sep='\t', header=None, names=['subject', 'object', 'relation'], engine='python')
    print('\tLoading FB15k training (valid file) data...')
    
    dataset = led.label_encode_dataset(entity, relation, data)

    # create node embeddings if none exists
    if not osp.exists("embeddings"):
        cne.create_node_embedding(dataset)
    embedded_nodes =  KeyedVectors.load_word2vec_format('embeddings/node_embedding.kv')

    dataset.x = torch.tensor(embedded_nodes.vectors, dtype=torch.float)
    edge_indexes = dataset.edge_index
    #data = GAE.split_edges(GAE, dataset)
    print(dataset)
    full_length = dataset.edge_index.shape[-1]
    train_index = torch.tensor(dataset.edge_index[:, 0:m.floor(full_length*0.7)], dtype=torch.long)
    train_attr_index = torch.tensor(dataset.edge_attr[0:m.floor(full_length*0.7)], dtype=torch.long)

    val_index = torch.tensor(dataset.edge_index[:, m.floor(full_length*0.7):m.floor(full_length*0.9)], dtype=torch.long)
    val_attr_index = torch.tensor(dataset.edge_attr[m.floor(full_length*0.7):m.floor(full_length*0.9)], dtype=torch.long)

    test_index = torch.tensor(dataset.edge_index[:, m.floor(full_length*0.9):], dtype=torch.long)
    test_attr_index = torch.tensor(dataset.edge_attr[m.floor(full_length*0.9):], dtype=torch.long)

    

    dataset.edge_index = torch.cat([train_index, val_index, test_index], dim=1)
    dataset.edge_attr = torch.cat([train_attr_index, val_attr_index, test_attr_index])

    dataset.edge_train_mask = torch.cat([torch.ones((train_index.size(-1))),
                                      torch.zeros((val_index.size(-1))),
                                      torch.zeros((test_index.size(-1)))], dim=0).byte()
    dataset.edge_val_mask = torch.cat([torch.zeros((train_index.size(-1))),
                                    torch.ones((val_index.size(-1))),
                                    torch.zeros((test_index.size(-1)))], dim=0).byte()
    dataset.edge_test_mask = torch.cat([torch.zeros((train_index.size(-1))),
                                     torch.zeros((val_index.size(-1))),
                                     torch.ones((test_index.size(-1)))], dim=0).byte()

    dataset.edge_train_attr_mask = torch.cat([torch.ones((train_attr_index.size(-1))),
                                      torch.zeros((val_attr_index.size(-1))),
                                      torch.zeros((test_attr_index.size(-1)))], dim=0).byte()
    dataset.edge_val_attr_mask = torch.cat([torch.zeros((train_attr_index.size(-1))),
                                    torch.ones((val_attr_index.size(-1))),
                                    torch.zeros((test_attr_index.size(-1)))], dim=0).byte()
    dataset.edge_test_attr_mask = torch.cat([torch.zeros((train_attr_index.size(-1))),
                                     torch.zeros((val_attr_index.size(-1))),
                                     torch.ones((test_attr_index.size(-1)))], dim=0).byte()

    dataset.edge_type = torch.zeros(((dataset.edge_index.size(-1)),)).long()

    dataset.batch = torch.zeros((1, dataset.num_nodes), dtype=torch.int64).view(-1)
    dataset.num_graphs = 1
    num_features = dataset.x.shape[-1] 
    num_relations = max(np.unique(dataset.edge_attr)) + 1
    return dataset, num_features, num_relations


def main(_args):
    print("@@@@@@@@@@@@@@@@ Multi-Relational Link Prediction @@@@@@@@@@@@@@@@")
    args = gap.parser.parse_args(_args)
    args.dataset = 'FB15K_2'
    data, num_features, num_relations = load_data()
    gap.tab_printer(data)
    print("\n=================== Run Trainer ===================\n")
    
    trainer.trainer(args, args.dataset, [data], [data], [data], transductive=True,
                    num_features=num_features, num_relations=num_relations, max_epoch=args.epochs,
                    num_node_class=0,
                    link_prediction=True)
    


if __name__ == '__main__':
    main(sys.argv[1:])
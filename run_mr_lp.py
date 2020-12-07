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
from torch_geometric.utils import structured_negative_sampling


ssl._create_default_https_context = ssl._create_unverified_context
def shuffle_dataset(dataset):
    df = pd.DataFrame([dataset.edge_index[0], dataset.edge_index[1], dataset.edge_type]).T
    df = df.sample(frac=1)
    edge_attributes = torch.tensor(list(df[2].values), dtype=torch.long)
    edge_index = torch.tensor([list(df[0].values), list(df[1].values)], dtype=torch.long)
    dataset.edge_type = edge_attributes
    dataset.edge_attr = edge_attributes
    dataset.edge_index = edge_index 
    return dataset

def train_val_test_split(dataset):
    train_size = int(np.floor(dataset.num_nodes*0.7))
    valid_size = int(np.floor(dataset.num_nodes*0.9))
    
    # edge_indexes
    dataset.train_pos_edge_index = dataset.edge_index.T[0:train_size].T
    dataset.val_pos_edge_index = dataset.edge_index.T[train_size : valid_size].T
    dataset.test_pos_edge_index = dataset.edge_index.T[valid_size:].T
    
    # relations
    dataset.train_edge_type = dataset.edge_type[0:train_size]
    dataset.val_edge_type = dataset.edge_type[train_size : valid_size]
    dataset.test_edge_type = dataset.edge_type[valid_size:]

    # negatives
    neg_train = structured_negative_sampling(dataset.train_pos_edge_index)
    dataset.train_neg_edge_index = torch.tensor([list(neg_train[0]), list(neg_train[2])], dtype=torch.long)

    neg_val = structured_negative_sampling(dataset.val_pos_edge_index)
    dataset.val_neg_edge_index = torch.tensor([list(neg_val[0]), list(neg_val[2])], dtype=torch.long)
    
    neg_test = structured_negative_sampling(dataset.test_pos_edge_index)
    dataset.test_neg_edge_index = torch.tensor([list(neg_test[0]), list(neg_test[2])], dtype=torch.long)
    
    return dataset

def load_data():
    entity_id = pd.read_csv('./data/FB15k/entities.txt', sep='\t', header=None, names=['entity', 'id'], engine='python')
    entity = entity_id['entity'].values

    relation_id = pd.read_csv('./data/FB15k/relations.txt', sep='\t', header=None, names=['relation', 'id'], engine='python')
    relation = relation_id['relation'].values

    print('\tLoading FB15k data...')
    # concat all data 
    name = ['subject', 'object', 'relation']

    train = pd.read_csv('./data/FB15k/train.txt', sep='\t', header=None, names=name, engine='python')
    valid = pd.read_csv('./data/FB15k/valid.txt', sep='\t', header=None, names=name, engine='python')
    test = pd.read_csv('./data/FB15k/test.txt', sep='\t', header=None, names=name, engine='python')
    data = pd.concat([train, valid])
    data = pd.concat([data, test])
    
    dataset, relations = led.label_encode_dataset(entity, relation, data)

    # create node embeddings if none exists
    if not osp.exists("embeddings"):
        print('No embeddings found. Creating Node2Vec embeddings...')
        cne.create_node_embedding(dataset)
    embedded_nodes =  KeyedVectors.load_word2vec_format('embeddings/node_embedding.kv')

    dataset.x = torch.tensor(embedded_nodes.vectors, dtype=torch.float)
    print(dataset)
    
    data = dataset
    data.edge_type = torch.LongTensor(relations) #torch.zeros(((data.edge_index.size(-1)),)).long()
    data.batch = torch.zeros((1, data.num_nodes), dtype=torch.int64).view(-1)
    data.num_graphs = 1
    num_features = dataset.x.shape[-1] 
    relation_dimension = len(np.unique(relations))
    print(f"no. unique relations: {relation_dimension}")
    print(f"no. edge_type size: {data.edge_type.size()}")
    print(f"no. relation size: {relations.shape}")
    print(f"edge_index size: {data.edge_index.size()}")
    print(f"min: {np.min(relations)}")
    print(f"min: {np.max(relations)}")

    # Shuffle and split
    data = shuffle_dataset(data)
    data = train_val_test_split(data)    

    data.edge_index = torch.cat([data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index], dim=1)

    data.edge_train_mask = torch.cat([torch.ones((data.train_pos_edge_index.size(-1))),
                                      torch.zeros((data.val_pos_edge_index.size(-1))),
                                      torch.zeros((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    data.edge_val_mask = torch.cat([torch.zeros((data.train_pos_edge_index.size(-1))),
                                    torch.ones((data.val_pos_edge_index.size(-1))),
                                    torch.zeros((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    data.edge_test_mask = torch.cat([torch.zeros((data.train_pos_edge_index.size(-1))),
                                     torch.zeros((data.val_pos_edge_index.size(-1))),
                                     torch.ones((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    

    return data, num_features, relation_dimension


def main(_args):
    print("@@@@@@@@@@@@@@@@ Multi-Relational Link Prediction @@@@@@@@@@@@@@@@")
    args = gap.parser.parse_args(_args)
    args.dataset = 'FB15K_1024_Hid'
    data, num_features, relation_dimension = load_data()
    gap.tab_printer(data)
    print("\n=================== Run Trainer ===================\n")
    # Auto run for pyCharm
    '''
    args.dropout = 0
    args.hidden = 1024
    args.l2 = 5e-4
    args.num_layers = 3
    args.cross_layer = False
    args.patience = 500
    args.residual = True
    args.residual_star = True
    args.epochs = 10
    args.device = 'cpu'
    '''
    trainer.trainer(args, args.dataset, [data], [data], [data], transductive=True,
                    num_features=num_features, relation_dimension=relation_dimension,
                    max_epoch=args.epochs, num_node_class=0,
                    link_prediction=True)
    


if __name__ == '__main__':
    main(sys.argv[1:])
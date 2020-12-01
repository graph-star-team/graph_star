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


ssl._create_default_https_context = ssl._create_unverified_context





def load_data():
    entity_id = pd.read_csv('./data/FB15k/entities.txt', sep='\t', header=None, names=['entity', 'id'], engine='python')
    entity = entity_id['entity'].values

    relation_id = pd.read_csv('./data/FB15k/relations.txt', sep='\t', header=None, names=['relation', 'id'], engine='python')
    relation = relation_id['relation'].values

    data = pd.read_csv('./data/FB15k/valid.txt', sep='\t', header=None, names=['subject', 'object', 'relation'], engine='python')
    print('\tLoading FB15k training (valid file) data...')
    
    dataset, relations = led.label_encode_dataset(entity, relation, data)

    # create node embeddings if none exists
    if not osp.exists("embeddings"):
        cne.create_node_embedding(dataset)
    embedded_nodes =  KeyedVectors.load_word2vec_format('embeddings/node_embedding.kv')

    dataset.x = torch.tensor(embedded_nodes.vectors, dtype=torch.float)
    print(dataset)
    
    '''data = GAE.split_edges(GAE, dataset)
    

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
    '''
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

    #print(f"no.  relations: {len(data.edge_type)}")

    

    return data, num_features, relation_dimension


def main(_args):
    print("@@@@@@@@@@@@@@@@ Multi-Relational Link Prediction @@@@@@@@@@@@@@@@")
    args = gap.parser.parse_args(_args)
    args.dataset = 'FB15K_2'
    data, num_features, relation_dimension = load_data()
    gap.tab_printer(data)
    print("\n=================== Run Trainer ===================\n")
    
    trainer.trainer(args, args.dataset, [data], [data], [data], transductive=True,
                    num_features=num_features, relation_dimension=relation_dimension,
                    max_epoch=args.epochs, num_node_class=0,
                    link_prediction=True)
    


if __name__ == '__main__':
    main(sys.argv[1:])
import os
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
import utils.create_relation_embedding as cre
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import structured_negative_sampling


ssl._create_default_https_context = ssl._create_unverified_context
def shuffle_dataset(dataset):
    df = pd.DataFrame([dataset.edge_index[0], dataset.edge_index[1], dataset.edge_type]).T
    df = df.sample(frac=1)
    edge_attributes = torch.tensor(list(df[2].values), dtype=torch.long)
    edge_index = torch.tensor([list(df[0].values), list(df[1].values)], dtype=torch.long)
    dataset.edge_type = edge_attributes
    dataset.edge_index = edge_index 
    return dataset

def train_val_test_split(dataset, val, test):
   
    
    # edge_indexes
    dataset.train_pos_edge_index = dataset.edge_index
    dataset.val_pos_edge_index = val.edge_index
    dataset.test_pos_edge_index = test.edge_index
    
    # relations
    dataset.train_edge_type = dataset.edge_type
    dataset.val_edge_type = val.edge_type
    dataset.test_edge_type = test.edge_type

    # negatives
    neg_train = structured_negative_sampling(dataset.train_pos_edge_index)
    dataset.train_neg_edge_index = torch.tensor([list(neg_train[0]), list(neg_train[2])], dtype=torch.long)

    neg_val = structured_negative_sampling(dataset.val_pos_edge_index)
    dataset.val_neg_edge_index = torch.tensor([list(neg_val[0]), list(neg_val[2])], dtype=torch.long)
    
    neg_test = structured_negative_sampling(dataset.test_pos_edge_index)
    dataset.test_neg_edge_index = torch.tensor([list(neg_test[0]), list(neg_test[2])], dtype=torch.long)
    
    return dataset

def load_data(hidden=64, dataset="FB15k"): #TODO - dataset choice as run_mr_lp argument
    entity_ids = pd.read_csv('./data/FB15k/entities.txt', sep='\t', header=None, names=['entity', 'id'], engine='python')
    relation_id = pd.read_csv('./data/FB15k/relations.txt', sep='\t', header=None, names=['relation', 'id'], engine='python')
    entities = entity_ids['entity'].values
    relations = relation_id['relation'].values


    # fit entity and relation encoder
    le_entity = LabelEncoder()
    le_entity.fit(entities)
    le_relation = LabelEncoder()
    le_relation.fit(relations)

    np.save(os.path.join('embeddings','le_relation_classes.npy'), le_relation.classes_)
    ''' TO LOAD SAVED ENCODER USE:
    self.le_relation = LabelEncoder()
    self.le_relation.classes_ = np.load(osp.join('embeddings','le_relation_classes.npy'), allow_pickle=True)
    '''
    all_relations = le_relation.transform(relations) # TODO : remove (use relations or embedded_relations directly istead?)


    print('\tLoading '+dataset+' data...')
    columns = {'FB15k' : ['head', 'tail', 'relation'], 'FB15k_237' : ['head', 'relation', 'tail']}

    train = pd.read_csv('./data/'+dataset+'/train.txt', sep='\t', header=None, names=columns[dataset], engine='python')
    valid = pd.read_csv('./data/'+dataset+'/valid.txt', sep='\t', header=None, names=columns[dataset], engine='python')
    test = pd.read_csv('./data/'+dataset+'/test.txt', sep='\t', header=None, names=columns[dataset], engine='python')
    
    dataset = led.label_encode_dataset(le_entity, le_relation, train)
    valid = led.label_encode_dataset(le_entity, le_relation, valid)
    test = led.label_encode_dataset(le_entity, le_relation, test)

    # create node embeddings if none exists
    if not osp.exists("embeddings"):
        print('No embeddings found. Creating Node2Vec embeddings...')
        cne.create_node_embedding(dataset)
    cre.create_relation_embedding(relations, le_relation, dimensions=hidden)
    embedded_nodes =  KeyedVectors.load_word2vec_format('embeddings/node_embedding.kv')
    embedded_relations = KeyedVectors.load_word2vec_format('embeddings/relation_embedding_le_'+str(hidden)+'.bin', binary=True)

    dataset.x = torch.tensor(embedded_nodes.vectors, dtype=torch.float)
    
    data = dataset
    data.batch = torch.zeros((1, data.num_nodes), dtype=torch.int64).view(-1)
    data.num_graphs = 1
    num_features = dataset.x.shape[-1] 
    relation_dimension = len(np.unique(all_relations))

    # Shuffle and split
    data = shuffle_dataset(data)
    data = train_val_test_split(data, valid, test)    
    
    data.edge_index = torch.cat([data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index], dim=1)
    data.edge_type = torch.cat([data.edge_type, valid.edge_type, test.edge_type], dim=0)
    print(data.edge_index)
    print(data.edge_type)
    data.edge_train_mask = torch.cat([torch.ones((data.train_pos_edge_index.size(-1))),
                                      torch.zeros((data.val_pos_edge_index.size(-1))),
                                      torch.zeros((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    data.edge_val_mask = torch.cat([torch.zeros((data.train_pos_edge_index.size(-1))),
                                    torch.ones((data.val_pos_edge_index.size(-1))),
                                    torch.zeros((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    data.edge_test_mask = torch.cat([torch.zeros((data.train_pos_edge_index.size(-1))),
                                     torch.zeros((data.val_pos_edge_index.size(-1))),
                                     torch.ones((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    
    print(f"no. unique relations: {relation_dimension}")
    print(f"no. relation size: {all_relations.shape}")
    print(f"size of training: {data.train_pos_edge_index.size()}")
    print(f"size of validation: {data.val_pos_edge_index.size()}")
    print(f"size of testing: {data.test_pos_edge_index.size()}")
    print(f"min rel: {np.min(all_relations)}")
    print(f"max rel: {np.max(all_relations)}")

    return data, num_features, relation_dimension, embedded_relations


def main(_args):
    print("@@@@@@@@@@@@@@@@ Multi-Relational Link Prediction @@@@@@@@@@@@@@@@")
    args = gap.parser.parse_args(_args)
    args.dataset = 'FB15k_237'
    data, num_features, relation_dimension, embedded_relations = load_data(hidden=args.hidden, dataset=args.dataset)
    #gap.tab_printer(data)
    
    trainer.trainer(args, args.dataset, [data], [data], [data], transductive=True,
                    num_features=num_features, relation_dimension=relation_dimension, relation_embeddings=embedded_relations,
                    num_epoch=args.epochs, num_node_class=0,
                    link_prediction=True)
    


if __name__ == '__main__':
    main(sys.argv[1:])
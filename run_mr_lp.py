import sys
import ssl
from os import path, mkdir
import numpy as np
import pandas as pd
import torch
import trainer
from torch_geometric.data import Data
import utils.gsn_argparse as gap
import utils.label_encode_dataset as led
import utils.create_node_embedding as cne
import utils.create_relation_embedding as cre
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

def train_val_test_split(dataset, train, val, test):
    # edge_indexes
    dataset.train_pos_edge_index = train.edge_index
    dataset.val_pos_edge_index = val.edge_index
    dataset.test_pos_edge_index = test.edge_index
    
    # relations
    dataset.train_edge_type = train.edge_type
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

def load_data(dataset, hidden=64, node_embedding_size=16, embedding_path='embeddings'):
    print('Loading '+dataset+' data...')
    columns = {'FB15k' : ['head', 'tail', 'relation'], 'FB15k_237' : ['head', 'relation', 'tail']}

    train = pd.read_csv('./data/'+dataset+'/train.txt', sep='\t', header=None, names=columns[dataset], engine='python')
    valid = pd.read_csv('./data/'+dataset+'/valid.txt', sep='\t', header=None, names=columns[dataset], engine='python')
    test = pd.read_csv('./data/'+dataset+'/test.txt', sep='\t', header=None, names=columns[dataset], engine='python')
    
    # needed for embedding all nodes across datasets (will use edge_index to split datasets)
    all_data = pd.concat([train, valid])
    all_data = pd.concat([all_data, test])
    all_data.drop_duplicates(inplace=True)

    entities = np.concatenate([all_data['head'].values, all_data['tail']])
    entities = np.unique(entities)
    relations = np.unique(all_data['relation'])

    # fit entity and relation encoder
    le_entity = LabelEncoder()
    le_entity.fit(entities)
    le_relation = LabelEncoder()
    le_relation.fit(relations)

    if not path.exists(embedding_path):
        mkdir(embedding_path)

    np.save(path.join(embedding_path,'le_relation_classes.npy'), le_relation.classes_)
    ''' TO LOAD SAVED ENCODER USE:
    self.le_relation = LabelEncoder()
    self.le_relation.classes_ = np.load(path.join('embeddings','le_relation_classes.npy'), allow_pickle=True)
    '''

    train = led.label_encode_dataset(le_entity, le_relation, train)
    valid = led.label_encode_dataset(le_entity, le_relation, valid)
    test = led.label_encode_dataset(le_entity, le_relation, test)

    # create node embeddings if none exists
    all_data = led.label_encode_dataset(le_entity, le_relation, all_data)

    cne.create_node_embedding(all_data, dataset, dimensions=node_embedding_size, workers=4)
    cre.create_relation_embedding(relations, le_relation, dataset, dimensions=hidden)
    embedded_nodes =  KeyedVectors.load_word2vec_format('embeddings/node_embedding_' +  dataset + '_' + str(node_embedding_size) + '.kv')
    embedded_relations = KeyedVectors.load_word2vec_format('embeddings/relation_embedding_le_' + dataset + '_' + str(hidden) + '.bin', binary=True)

    # need to sort to get correct indexing
    sorted_embedding = []
    for i in range(0, len(embedded_nodes.vectors)):
        sorted_embedding.append(embedded_nodes.get_vector(str(i)))
    all_data.x = torch.tensor(sorted_embedding, dtype=torch.float)
    
    sorted_embedding = []
    for i in range(0, len(embedded_relations.vectors)):
        sorted_embedding.append(embedded_relations.get_vector(str(i)))
    embedded_relations = torch.tensor(sorted_embedding, dtype=torch.float)

    all_data.batch = torch.zeros((1, all_data.num_nodes), dtype=torch.int64).view(-1)

    all_data.num_graphs = 1
    num_features = all_data.x.shape[-1] 
    num_relations = len(np.unique(relations))

    # Shuffle and split
    #data = shuffle_dataset(all_data) why?
    data = train_val_test_split(all_data, train, valid, test)    
    
    data.edge_index = torch.cat([data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index], dim=1)
    data.edge_type = torch.cat([train.edge_type, valid.edge_type, test.edge_type], dim=0)
    
    print(f"no. unique relations: {num_relations}")
    print(f"size of training: p-{data.train_pos_edge_index.size()}, n-{data.train_neg_edge_index.size()}")
    print(f"size of validation: p-{data.val_pos_edge_index.size()}, n-{data.val_neg_edge_index.size()}")
    print(f"size of testing: p-{data.test_pos_edge_index.size()}, n-{data.test_neg_edge_index.size()}")

    return data, num_features, num_relations, embedded_relations


def main(_args):
    print("@@@@@@@@@@@@@@@@ Multi-Relational Link Prediction @@@@@@@@@@@@@@@@")
    args = gap.parser.parse_args(_args)
    data, num_features, num_relations, embedded_relations = load_data(hidden=args.hidden, dataset=args.dataset)
    #gap.tab_printer(data)
    
    # python run_mr_lp.py --dropout=0 --hidden=128 --l2=5e-4 --num_layers=3 --cross_layer=False --patience=200 --residual=True --residual_star=True --dataset=FB15k_237 --device=cpu --epochs=2
    trainer.trainer(args, args.dataset, data,
                    num_features=num_features, num_relations=num_relations, relation_embeddings=embedded_relations,
                    num_epoch=args.epochs) 
    


if __name__ == '__main__':
    main(sys.argv[1:])
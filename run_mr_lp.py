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
import numpy as np
import pandas as pd
from sklearn import preprocessing

ssl._create_default_https_context = ssl._create_unverified_context

# label encoding (entity->id & relation->id)
name = ['entity', 'id']
entity_id = pd.read_csv('./data/FB15k/entities.txt', sep='\t', header=None, names=name, engine='python')
entity = entity_id['entity'].values.tolist()
le_entity = preprocessing.LabelEncoder()
le_entity.fit(entity)

name = ['relation', 'id']
relation_id = pd.read_csv('./data/FB15k/relations.txt', sep='\t', header=None, names=name, engine='python')
relation = relation_id['relation'].values.tolist()
le_relation = preprocessing.LabelEncoder()
le_relation.fit(relation)


def load_data():
    name = ['subject', 'object', 'relation']
    data = pd.read_csv('./data/FB15k/valid.txt', sep='\t', header=None, names=name, engine='python')
    print('\tLoading FB15k training (valid file) data...')

    subjects = data['subject'].values.tolist()
    objects = data['object'].values.tolist()
    relations = data['relation'].values.tolist()

    # string list to int array using LabelEncoder
    subjects = np.array(le_entity.transform(subjects))
    objects = np.array(le_entity.transform(objects))
    relations = np.array(le_relation.transform(relations))

    train_set_raw = np.concatenate([subjects.reshape(-1, 1), relations.reshape(-1, 1), objects.reshape(-1, 1)], axis=1)
    n_pos_samples = len(relations)

    del name, data, subjects, objects, relations, n_pos_samples

    # entity_pair = [[subject, object]] y = [[relation]] edge_index = [[subject],[object]]
    entity_pairs = torch.tensor(train_set_raw[:, 0:3:2], dtype=torch.float)
    y = torch.tensor(train_set_raw[:, 1], dtype=torch.float)
    edge_index = torch.tensor([train_set_raw[:,0], train_set_raw[:,2]], dtype=torch.long)

    
    dataset = Data(x=entity_pairs, y=y, edge_index=edge_index)
    data = GAE.split_edges(GAE, dataset)
    del entity_pairs, y, dataset, train_set_raw
    
    data.train_pos_edge_index = gutils.to_undirected(data.train_pos_edge_index)
    data.val_pos_edge_index = gutils.to_undirected(data.val_pos_edge_index)
    data.test_pos_edge_index = gutils.to_undirected(data.test_pos_edge_index)

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

    data.edge_type = torch.zeros(((data.edge_index.size(-1)),)).long()

    data.batch = torch.zeros((1, data.num_nodes), dtype=torch.int64).view(-1)
    data.num_graphs = 1
    num_features=2
    return data, num_features


def main(_args):
    print("@@@@@@@@@@@@@@@@ Multi-Relational Link Prediction @@@@@@@@@@@@@@@@")
    args = gap.parser.parse_args(_args)
    args.dataset = 'FB15K'
    data, num_features = load_data()
    gap.tab_printer(data)
    print("\n=================== Run Trainer ===================\n")
    
    trainer.trainer(args, args.dataset, [data], [data], [data], transductive=True,
                    num_features=num_features, max_epoch=args.epochs,
                    num_node_class=0,
                    link_prediction=True)
    


if __name__ == '__main__':
    main(sys.argv[1:])
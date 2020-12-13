from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric.data import Data
import numpy as np
import torch


def label_encode_dataset(all_entities, all_relations, data):
    head = data['head'].values
    tail = data['tail'].values
    relations = data['relation'].values
    
    # fit entity encoder
    le_entity = LabelEncoder()
    le_entity.fit(all_entities)

    # fit relationship encoder
    le_relation = LabelEncoder()
    le_relation.fit(all_relations.reshape(-1, 1))

    # string list to int array using LabelEncoder on complete data set
    heads = le_entity.transform(head)
    tails = le_entity.transform(tail)
    relations = le_relation.transform(relations)
    all_relations = le_relation.transform(all_relations.reshape(-1, 1))
    # encode subsample (change range to 0-N)
    '''
    le_entity2 = LabelEncoder().fit(np.append(subjects,objects))
    le_rel = LabelEncoder().fit(relations)


    subjects = le_entity2.transform(subjects)
    objects = le_entity2.transform(objects)
    relations = le_rel.transform(relations)
    '''
    
    edge_attributes = torch.tensor(relations, dtype=torch.long)
    edge_index = torch.tensor([heads, tails], dtype=torch.long)
    unique_entities = torch.tensor(np.unique(edge_index.reshape(edge_index.shape[-1]*2, 1)), dtype=torch.float)

    return Data(x=unique_entities, edge_type=edge_attributes, edge_index=edge_index), all_relations
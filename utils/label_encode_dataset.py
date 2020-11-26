from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric.data import Data
import numpy as np
import torch


def label_encode_dataset(all_entities, all_relations, data):
    subjects = data['subject'].values
    objects = data['object'].values
    relations = data['relation'].values
    
    # fit entity encoder
    le_entity = LabelEncoder()
    le_entity.fit(all_entities)

    # fit relationship encoder
    ohe_relation = LabelEncoder()
    ohe_relation.fit(all_relations.reshape(-1, 1))

    # string list to int array using LabelEncoder on complete data set
    subjects = le_entity.transform(subjects)
    objects = le_entity.transform(objects)
    relations = ohe_relation.transform(relations)
    
    # encode subsample (change range to 0-N)
    le_entity2 = LabelEncoder().fit(np.append(subjects,objects))

    subjects = le_entity2.transform(subjects)
    objects = le_entity2.transform(objects)
    
    edge_attributes = torch.tensor(relations, dtype=torch.float)
    edge_index = torch.tensor([subjects, objects], dtype=torch.long)
    unique_entities = torch.tensor(np.unique(edge_index.reshape(edge_index.shape[-1]*2, 1)), dtype=torch.float)

    return Data(x=unique_entities, edge_attr=edge_attributes, edge_index=edge_index), relations
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import numpy as np
import torch


def label_encode_dataset(le_entity, le_relation, data):
    head = data['head'].values
    tail = data['tail'].values
    relations = data['relation'].values

    # string list to int array using LabelEncoder on complete data set
    heads = le_entity.transform(head)
    tails = le_entity.transform(tail)
    relations = le_relation.transform(relations)

    
    edge_attributes = torch.tensor(relations, dtype=torch.long)
    edge_index = torch.tensor([heads, tails], dtype=torch.long)
    unique_entities = torch.tensor(np.unique(edge_index.reshape(edge_index.shape[-1]*2, 1)), dtype=torch.float)

    return Data(x=unique_entities, edge_type=edge_attributes, edge_index=edge_index)
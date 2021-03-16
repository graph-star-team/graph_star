from torch_geometric.utils import to_networkx
from node2vec import Node2Vec
import os

def create_node_embedding(dataset, dataset_name, path="embeddings", dimensions=16, walk_length=15, num_walks=20, workers=1, window=10, min_count=1, batch_words=3):
    embedding_name = 'node_embedding_' + dataset_name + '_' + str(dimensions) + ".kv"
    if not os.path.exists(os.path.join(path, embedding_name)):
        print('Embeddings not found. Creating node embeddings...')
        G = to_networkx(dataset)
    
        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)  # Use temp_folder for big graphs
        
        # Embed nodes
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)  
        # Any keywords acceptable by gensim.Word2Vec can be passed, 
        # `dimensions` and `workers` are automatically passed
        # (from the Node2Vec constructor)
        
        # Save embeddings for later use
        model.wv.save_word2vec_format(os.path.join(path, embedding_name))
        print(f"Saved embedding and model in the {path} folder")

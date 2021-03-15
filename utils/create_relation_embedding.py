import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec 
from sklearn.preprocessing import LabelEncoder
from smart_open import open

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

def create_relation_embedding(relations, le_relation, path="embeddings", gensimCorpus="text8", dimensions=256):
    stripped_relations = []
    for relation in relations:
        relation = relation.replace('.','')
        stripped_relation = relation.split('/')
        stripped_relations.append(stripped_relation[1:])

    kv_path = 'relation_embedding_'+str(dimensions)+'.bin'
    kv_path_le = 'relation_embedding_le_'+str(dimensions)+'.bin'
    if not os.path.exists(path):
        os.mkdir(path)
    
    if not os.path.exists(os.path.join(path, kv_path)):
        print('Embeddings not found. Creating relation embeddings...')
        corpus = api.load(gensimCorpus)
        model = Word2Vec(corpus, min_count=1, size=dimensions)
        model.build_vocab(stripped_relations, update=True)
        model.train(stripped_relations, total_examples=len(stripped_relations), epochs=3)
        keyedVectors = {}
        keyedVectors_le = {}
        lable_encoded_relations = le_relation.transform(relations)
        for i in range(len(stripped_relations)):
            relation_embedding = np.array([model.wv[word] for word in stripped_relations[i]]).mean(axis=0)
            keyedVectors[relations[i]] = relation_embedding  
            keyedVectors_le[str(lable_encoded_relations[i])] = relation_embedding
        
        save_word2vec_format(binary=True, fname=os.path.join(path, kv_path), vocab=keyedVectors, vector_size=dimensions)
        save_word2vec_format(binary=True, fname=os.path.join(path, kv_path_le), vocab=keyedVectors_le, vector_size=dimensions)


def save_word2vec_format(fname, vocab, vector_size, binary=True):    
    total_vec = len(vocab)
    with open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in tqdm(vocab.items()):
            if binary:
                row = row.astype(np.float32)
                fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))



def main():
    relation_id = pd.read_csv('./data/FB15k/relations.txt', sep='\t', header=None, names=['relation', 'id'], engine='python')
    relations = relation_id['relation'].values
    create_relation_embedding(relations)

if __name__ == '__main__':
    main()
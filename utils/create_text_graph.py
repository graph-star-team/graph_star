from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import sys, ssl
import re

import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
from sklearn.preprocessing import OneHotEncoder

# Disable ssl certificte verify for some use cases
ssl._create_default_https_context = ssl._create_unverified_context

# Some of codes in this file are from text-gcn

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
    
def remove_words(dataset):
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    print(stop_words)

    doc_content_list = []
    f = open('../data/TC/corpus/' + dataset + '.txt', 'rb')
    # f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))
    f.close()


    word_freq = {}  # to remove rare words

    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_docs = []
    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        doc_words = []
        for word in words:
            # word not in stop_words and word_freq[word] >= 5
            if dataset == 'mr':
                doc_words.append(word)
            elif word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        #if doc_str == '':
            #doc_str = temp
        clean_docs.append(doc_str)

    clean_corpus_str = '\n'.join(clean_docs)

    f = open('../data/TC/corpus/' + dataset + '.clean.txt', 'w')
    f.write(clean_corpus_str)
    f.close()

    #dataset = '20ng'
    min_len = 10000
    aver_len = 0
    max_len = 0 

    f = open('../data/TC/corpus/' + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    f.close()
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))
    
def loadWord2Vec(dataset, use_pretrain_vec=True, vocab=None, emb_sz=200):
    """Read Word Vectors"""
    if use_pretrain_vec:
        if dataset == '20ng':
            filename = '../data/TC/' + dataset + '_word_vectors_0.txt'
        else:
            filename = '../data/TC/' + dataset + '_word_vectors.txt'
        vocab = []
        embd = []
        word_vector_map = {}
        with open(filename, 'r') as file:
            for line in file.readlines():
                row = line.strip().split(' ')
                if(len(row) > 2):
                    vocab.append(row[0])
                    vector = row[1:]
                    length = len(vector)
                    for i in range(length):
                        vector[i] = float(vector[i])
                    embd.append(vector)
                    word_vector_map[row[0]] = vector
        print('Loaded Word Vectors!')
    else:
        embd = []
        word_vector_map = {}
        word_vectors = np.random.uniform(-0.01, 0.01, (len(vocab), emb_sz))
        for word, vec in zip(vocab, word_vectors):
            embd.append(list(vec))
            word_vector_map[word] = list(vec)
    return vocab, embd, word_vector_map
    
def split_train_test(dataset):
    # shulffing
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    f = open('../data/TC/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()
    
    return doc_name_list, doc_train_list, doc_test_list

def get_doc_content(dataset):
    doc_content_list = []
    f = open('../data/TC/corpus/' + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    return doc_content_list

def write_train_test_ids(doc_name_list, doc_train_list, doc_test_list, dataset):
    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)

    train_ids_str = '\n'.join(str(index) for index in train_ids)
    f = open('../data/TC/' + dataset + '.train.index', 'w')
    f.write(train_ids_str)
    f.close()

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)

    test_ids_str = '\n'.join(str(index) for index in test_ids)
    f = open('../data/TC/' + dataset + '.test.index', 'w')
    f.write(test_ids_str)
    f.close()

    ids = train_ids + test_ids
    print(len(ids))
    return ids, train_ids, test_ids

def create_shuffle_file(dataset, ids, doc_name_list, doc_content_list):
    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for _id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(_id)])
        shuffle_doc_words_list.append(doc_content_list[int(_id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    f = open('../data/TC/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_name_str)
    f.close()

    f = open('../data/TC/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()
    return shuffle_doc_name_list, shuffle_doc_words_list

def build_vocab(dataset, shuffle_doc_words_list):
    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    word_doc_list = {}

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    f = open('../data/TC/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    return vocab, word_id_map

def get_label_list(shuffle_doc_name_list, dataset):
    # label list
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    f = open('../data/TC/corpus/' + dataset + '_labels.txt', 'w')
    f.write(label_list_str)
    f.close()
    
    return label_list

def split_train_val(dataset, train_ids, shuffle_doc_name_list):
    # select 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('../data/TC/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()
    
    return real_train_doc_names, real_train_size

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def get_all_nodes_in_all_graph_20ng(word_vector_map, shuffle_doc_words_list, max_length=1024):
    '''repeat words with different context are treated as same nodes, only used for 20ng
    '''
    node_attrs = []
    graph_indicator = []
    for doc_ind, doc in enumerate(shuffle_doc_words_list):
#         doc_words = list(set(doc.split()))
#         if len(doc_words) > 600:
#             continue
        doc_words = list(set(doc.split()[: max_length]))
        doc_node_attr = []
        for word in doc_words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
            doc_node_attr.append(word_vector)
            graph_indicator.append(doc_ind + 1)   
        doc_node_attr = np.array(doc_node_attr)
#         doc_node_attr = preprocess_features(doc_node_attr)
        node_attrs.append(doc_node_attr)
    node_attrs_final = np.array([it for _doc in node_attrs for it in _doc])
    graph_indicator = np.array(graph_indicator)
    return node_attrs_final, graph_indicator

def get_all_nodes_in_all_graph(word_vector_map, shuffle_doc_words_list, max_length=1024):
    '''repeat words with different context are treated as different nodes
    '''
    node_attrs = []
    graph_indicator = []
    for doc_ind, doc in enumerate(shuffle_doc_words_list):
        doc_words = doc.split()[: max_length]
        doc_node_attr = []
        for word in doc_words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
            graph_indicator.append(doc_ind + 1)         
            doc_node_attr.append(word_vector)
        doc_node_attr = np.array(doc_node_attr)
#         doc_node_attr = preprocess_features(doc_node_attr)
        node_attrs.append(doc_node_attr)
    node_attrs_final = np.array([it for _doc in node_attrs for it in _doc])
    graph_indicator = np.array(graph_indicator)
    return node_attrs_final, graph_indicator
    
def get_graph_labels(shuffle_doc_name_list, shuffle_doc_words_list, label_list):
    g_lab = []
    for doc, doc_word in zip(shuffle_doc_name_list, shuffle_doc_words_list):
        doc_words = list(set(doc_word.split()))
        temp = doc.split('\t')
        label = temp[2]
#         one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
#         one_hot[label_index] = 1
        g_lab.append(label_index + 1)
    g_lab = np.array(g_lab)
    return g_lab

def get_doc_windows(shuffle_doc_words_list, window_size=20):
    doc_windows = []

    for doc_words in shuffle_doc_words_list:
        doc_win = []
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            doc_win.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                doc_win.append(window)
        doc_windows.append(doc_win)
        
    print('finished to cal doc window')
    print('length of docs is: ' + str(len(doc_windows)))
    
    return doc_windows

def cal_big_adj_20ng(shuffle_doc_words_list, graph_indicator, max_num_nodes, window_size=10, max_length=1024):
    '''repeat words with different context are treated as same nodes, only used for 20ng
    '''
    count_n = 0
    adjs = []
    row = []
    col = []
    weight = []
    for i, doc_word in enumerate(shuffle_doc_words_list):
        doc_wind = []
        words = doc_word.split()[: max_length]
        length = len(words)
        doc_words = list(set(words))
        doc_len = len(doc_words)
            
        if length <= window_size:
            doc_wind.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                doc_wind.append(window)
                
        _tmp_word_id_map = {doc_words[ind]: ind + 1 for ind in range(doc_len)}
        word_pair_count = {}

        for window in doc_wind:
            for ind_x in window:
                for ind_y in window:
                    word_i_id = _tmp_word_id_map[ind_x]
                    word_j_id = _tmp_word_id_map[ind_y]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                        row.append(word_i_id + count_n)
                        col.append(word_j_id + count_n)
                        weight.append(1)
        count_n += doc_len           
        if i % 1000 == 0:
            print('finished to process: ' + str(i))
            print('unique word length of this doc is ' + str(doc_len))
            print('length of this doc window is ' + str(len(doc_wind)))     
            print('num of edges in the doc is ' + str(len(weight)))

    return row, col

def cal_big_adj(shuffle_doc_words_list, graph_indicator, max_num_nodes, window_size=10, max_length=1024):
    '''repeat words with different context are treated as different nodes
    '''
    count_n = 0
    adjs = []
    row = []
    col = []
    weight = []
    for i, doc_word in enumerate(shuffle_doc_words_list):
        doc_wind = []
        words = doc_word.split()[: max_length]
        length = len(words)
        doc_words = words
        doc_len = len(doc_words)
            
        if length <= window_size:
            doc_wind.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                doc_wind.append(window)
                
        for ind_x, x_word in enumerate(doc_words):
            for ind_y, y_word in enumerate(doc_words):
                if ind_x == ind_y:
                    continue
                repeat = False
                for wind in doc_wind:
                    if repeat:
                        continue
                    if (x_word in wind) and (y_word in wind):
                        repeat = True
                        row.append(ind_x + count_n + 1)
                        col.append(ind_y + count_n + 1)
                        weight.append(1)
                
        count_n += doc_len           
        if i % 1000 == 0:
            print('finished to process: ' + str(i))
            print('unique word length of this doc is ' + str(doc_len))
            print('length of this doc window is ' + str(len(doc_wind)))     
            print('num of edges in the doc is ' + str(len(weight)/2))
    return row, col

def get_max_num_node(shuffle_doc_words_list, max_length, use_repeat):
    maxs = 0
    for it in shuffle_doc_words_list:
        words = it.split()[: max_length]
        if use_repeat:
            words = list(set(words))
        length = len(words)
        if length > maxs:
            maxs = length
    return maxs

def get_combined_result(rows, cols):
    count = 0
    test_row = []
    test_col = []
    for single_row, single_col in zip(rows, cols):
        for it1, it2 in zip(single_row, single_col):
            test_row.append(it1 + count)
            test_col.append(it2 + count)
        count += np.max([np.max(single_row), np.max(single_col)])
    return test_row, test_col

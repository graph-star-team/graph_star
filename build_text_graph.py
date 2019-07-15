from utils.create_text_graph import *
from concurrent.futures import ThreadPoolExecutor, wait, as_completed, ProcessPoolExecutor
import time
import random

max_workers = 20  # number of workers used to generate text graph
dataset = 'R52'  # should be in ['R8', 'R52', 'mr', 'ohsumed', '20ng']
use_word_feat = True  # whether to use pretrained word vector, default should be true
max_length = 3072  # max length for the doc, 3072 is longer than the maxium length of mr, R8, R52, Ohsumed, but truncate for 20ng
word_embdz_size = 200  # only valid when use_word_feat is False
wind_size = 10  # sliding window size
use_repeat_word = False  # repeat words with different context are treated as same/different nodes, should be True for 20ng, and False for others

output_dir = './data/' + dataset + '/raw/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def main():
    try:
        remove_words(dataset)
    except FileNotFoundError:
        print('Please download all datasets from https://github.com/yao8839836/text_gcn/tree/master/data')
    #         url_add = 'https://github.com/yao8839836/text_gcn/tree/master/data'
    #         from torch_geometric.data import download_url
    #         download_url(url_add, '../data/TC')
    doc_name_list, doc_train_list, doc_test_list = split_train_test(dataset)
    doc_content_list = get_doc_content(dataset)
    ids, train_ids, test_ids = write_train_test_ids(doc_name_list, doc_train_list, doc_test_list, dataset)
    train_size = len(train_ids)
    shuffle_doc_name_list, shuffle_doc_words_list = create_shuffle_file(dataset, ids, doc_name_list, doc_content_list)
    vocab, word_id_map = build_vocab(dataset, shuffle_doc_words_list)
    vocab_size = len(vocab)
    label_list = get_label_list(shuffle_doc_name_list, dataset)
    _, embd, word_vector_map = loadWord2Vec(dataset, use_word_feat, vocab, word_embdz_size)
    word_embeddings_dim = len(embd[0])
    real_train_doc_names, real_train_size = split_train_val(dataset, train_ids, shuffle_doc_name_list)
    print('Training size is: %d, Val size is: %d' % (real_train_size, train_size - real_train_size))
    print('label list is: ')
    print(label_list)
    print('Number of vocab is: %d' % (len(word_vector_map)))
    print('The dim of word is: %d' % (word_embeddings_dim))

    if use_repeat_word:
        node_attrs, graph_indicator = get_all_nodes_in_all_graph_20ng(word_vector_map, shuffle_doc_words_list,
                                                                      max_length)
        graph_labels = get_graph_labels(shuffle_doc_name_list, shuffle_doc_words_list, label_list)
        max_num_nodes = get_max_num_node(shuffle_doc_words_list, max_length, use_repeat_word)
        print('use repeat word')
        print('max num nodes is ' + str(max_num_nodes))
        print('total num of nodes is ' + str(len(node_attrs)))
    else:
        node_attrs, graph_indicator = get_all_nodes_in_all_graph(word_vector_map, shuffle_doc_words_list, max_length)
        graph_labels = get_graph_labels(shuffle_doc_name_list, shuffle_doc_words_list, label_list)
        max_num_nodes = get_max_num_node(shuffle_doc_words_list, max_length, use_repeat_word)
        print('do not use repeat word')
        print('max num nodes is ' + str(max_num_nodes))
        print('total num of nodes is ' + str(len(node_attrs)))

    split_step = int(len(shuffle_doc_words_list) / max_workers)
    pool = ProcessPoolExecutor(max_workers=max_workers)

    index_list = list(range(0, len(shuffle_doc_words_list), split_step))
    print(index_list)

    res_list = []
    for ind, it in enumerate(index_list):
        if ind != len(index_list) - 1:
            print(index_list[ind], index_list[ind + 1])
            if use_repeat_word:
                future = pool.submit(cal_big_adj_20ng, shuffle_doc_words_list[index_list[ind]: index_list[ind + 1]],
                                     graph_indicator, max_num_nodes, wind_size, max_length)
            else:
                future = pool.submit(cal_big_adj, shuffle_doc_words_list[index_list[ind]: index_list[ind + 1]],
                                     graph_indicator, max_num_nodes, wind_size, max_length)
        else:
            print(index_list[ind])
            if use_repeat_word:
                future = pool.submit(cal_big_adj_20ng, shuffle_doc_words_list[index_list[ind]:], graph_indicator,
                                     max_num_nodes, wind_size, max_length)
            else:
                future = pool.submit(cal_big_adj, shuffle_doc_words_list[index_list[ind]:], graph_indicator,
                                     max_num_nodes, wind_size, max_length)
        res_list.append(future)

    print('start')
    wait(res_list)
    print('end')
    rows, cols = [], []
    for f in res_list:
        #     print(f.result())  # 查看task1返回的结果
        rows.append(f.result()[0])
        cols.append(f.result()[1])

    test_row, test_col = get_combined_result(rows, cols)

    with open('./data/' + dataset + '/raw/' + dataset + '_graph_labels.txt', 'w') as g_lab:
        for it in graph_labels:
            g_lab.write(str(it) + '\n')

    with open('./data/' + dataset + '/raw/' + dataset + '_graph_indicator.txt', 'w') as g_ind:
        for it in graph_indicator:
            g_ind.write(str(it) + '\n')

    with open('./data/' + dataset + '/raw/' + dataset + '_A.txt', 'w') as g_A:
        for ind, (r, c) in enumerate(zip(test_row, test_col)):
            g_A.write(str(r) + ', ' + str(c) + '\n')

    with open('./data/' + dataset + '/raw/' + dataset + '_node_attributes.txt', 'w') as g_node_att:
        for ind, it in enumerate(node_attrs):
            _it = [str(f) for f in it]
            f_str = ', '.join(_it)
            g_node_att.write(f_str + '\n')


if __name__ == '__main__':
    main()

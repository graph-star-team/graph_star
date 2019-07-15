import numpy as np
import os
import torch
import random

random.seed(31415926)


def load_data(filepath):
    bert_encode_res = np.load(os.path.join(filepath, "split_bert_large_encode_res.npy"))  # 25000,768
    y = np.load(os.path.join(filepath, "split_y.npy"))  # 25000
    edge = np.load(os.path.join(filepath, "split_edge.npy"))  # 2,edge_num*2
    bug_edge = np.load(os.path.join(filepath, "split_bug_edge.npy"))  # 2,edge_num*2
    datas = []
    for x, y, e, eb in zip(bert_encode_res, y, edge, bug_edge):
        x = np.array([_.tolist() for _ in x], dtype=np.float)
        y = np.array(y, dtype=np.long)

        if len(e) == 0:
            e = np.empty((0, 2), dtype=np.long).transpose()
            eb = np.empty((0, 2), dtype=np.long).transpose()
        else:
            e = np.array(e).transpose()
            eb = np.array(eb).transpose()
        datas.append((x, y, e, eb))
    random.shuffle(datas)

    max_node_num = 2000

    def empty():
        x = np.empty((0, 1024), dtype=np.float)
        y = np.empty(0, dtype=np.long)
        e = np.empty((0, 2), dtype=np.long).transpose()
        eb = np.empty((0, 2), dtype=np.long).transpose()
        return x, y, e, eb

    new_res = []
    n_x, n_y, n_e, n_eb = empty()
    for x, y, e, eb in datas:
        if len(n_x) + len(x) > max_node_num:
            new_res.append((n_x, n_y, n_e, n_eb))
            n_x, n_y, n_e, n_eb = empty()
        if len(e) > 0:
            e = e + len(n_x)
            eb = eb + len(n_x)
            n_e = np.concatenate((n_e, e), axis=1)
            n_eb = np.concatenate((n_eb, eb), axis=1)
        n_x = np.concatenate((n_x, x), axis=0)
        n_y = np.concatenate((n_y, y), axis=0)
    if len(n_x) > 0:
        new_res.append((n_x, n_y, n_e, n_eb))
    # print(new_res)
    xx = []
    yy = []
    ee = []
    eebb = []
    for x, y, e, eb in new_res:
        xx.append(x)
        yy.append(y)
        ee.append(e.transpose())
        eebb.append(eb.transpose())
    np.save(os.path.join(filepath, "split_2k_bert_large_encode_res.npy"), xx)
    np.save(os.path.join(filepath, "split_2k_edge.npy"), ee)
    np.save(os.path.join(filepath, "split_2k_bug_edge.npy"), eebb)
    np.save(os.path.join(filepath, "split_2k_y.npy"), yy)


load_data("../data/aclImdb/train")
load_data("../data/aclImdb/test")

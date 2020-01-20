import numpy as np
import os
import torch
import random
from bert_serving.client import BertClient

random.seed(31415926)

def build_imdb_npy(imdb_path):
    for file_path in [os.path.join(imdb_path, "train"),
                      os.path.join(imdb_path, "test")]:
        txt_list = []
        y = []
        for label in ["pos", "neg"]:
            train_path = os.path.join(file_path, label)

            for fname in os.listdir(train_path):
                f = open(os.path.join(train_path, fname))
                txt = ""
                for l in f.readlines():
                    txt += (l + " ")
                txt_list.append(txt)
                if label == "pos":
                    y.append(1)
                else:
                    y.append(0)
        y = np.array(y)
        bc = BertClient()
        res = bc.encode(txt_list)

        np.save(os.path.join(file_path, "bert_large_encode_res.npy"), res)
        np.save(os.path.join(file_path, "y.npy"), y)

        # res = np.load(os.path.join(file_path, "all_bert_fine_tuning_encode_res.npy"))
        # y = np.load(os.path.join(file_path, "all_y.npy"))

        topic_dic = dict()
        lines = []
        f = open(os.path.join(file_path, "urls_pos.txt"))
        lines.extend([x[26:35] for x in f.readlines()])
        f = open(os.path.join(file_path, "urls_neg.txt"))
        lines.extend([x[26:35] for x in f.readlines()])

        s_edge = []
        s_bug_edge = []
        s_be = []
        s_y = []
        t_idx = 0
        for idx, id in enumerate(lines):
            if id not in topic_dic:
                topic_dic[id] = len(topic_dic)
                s_edge.append([])
                s_bug_edge.append([])
                s_be.append([res[idx]])
                s_y.append([y[idx]])
                # t_idx += 1
            else:
                t_idx = topic_dic[id]
                new_idx = len(s_be[t_idx])
                for i in range(len(s_be[t_idx])):
                    s_edge[t_idx].append([i, new_idx])
                    s_edge[t_idx].append([new_idx, i])
                s_bug_edge[t_idx].append([0, new_idx])
                s_bug_edge[t_idx].append([new_idx, 0])

                s_be[t_idx].append(res[idx])
                s_y[t_idx].append(y[idx])
        np.save(os.path.join(file_path, "split_bert_large_encode_res.npy"), s_be)
        np.save(os.path.join(file_path, "split_edge.npy"), s_edge)
        np.save(os.path.join(file_path, "split_bug_edge.npy"), s_bug_edge)
        np.save(os.path.join(file_path, "split_y.npy"), s_y)

def load_data(filepath):
    bert_encode_res = np.load(os.path.join(filepath, "split_bert_large_encode_res.npy"),allow_pickle=True)  # 25000,768
    y = np.load(os.path.join(filepath, "split_y.npy"),allow_pickle=True)  # 25000
    edge = np.load(os.path.join(filepath, "split_edge.npy"),allow_pickle=True)  # 2,edge_num*2
    bug_edge = np.load(os.path.join(filepath, "split_bug_edge.npy"),allow_pickle=True)  # 2,edge_num*2
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

imdb_path = "/mnt/nas1/NLP/public_dataset/TC/imdb/aclImdb"
build_imdb_npy(imdb_path)
load_data(os.path.join(imdb_path,"train"))
load_data(os.path.join(imdb_path,"test"))

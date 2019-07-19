import time
import torch
import numpy as np
import os
import utils.tensorboard_writer as tw
import utils.gsn_argparse as gap
from utils.gsn_argparse import tab_printer

from torch.optim.lr_scheduler import ReduceLROnPlateau
from multiprocessing import Process, Queue
from module.graph_star import GraphStar


def get_edge_info(data, type):
    attr = "edge_" + type + "_mask"
    edge_index = data.edge_index[:, getattr(data, attr)] if hasattr(data, attr) else data.edge_index
    edge_type = None
    if hasattr(data, "edge_type"):
        edge_type = data.edge_type[getattr(data, attr)] if hasattr(data, attr) else data.edge_type
    return edge_index, edge_type


train_neg_sampling_queue = None
test_neg_sampling_queue = None
val_neg_sampling_queue = None


def train_transductive(model, optimizer, loader, device, node_classification, node_multi_label, graph_classification,
                       graph_multi_label, link_prediction, mode="train", cal_mrr_score=False):
    global train_neg_sampling_queue, test_neg_sampling_queue, val_neg_sampling_queue
    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_node = 0
    node_acc_count = 0
    total_graph = 0
    graph_acc_count = 0
    lp_auc = lp_ap = None
    data_count = 0
    for data in loader:
        data_count += data.num_graphs
        num_graphs = data.num_graphs
        data = data.to(device)
        optimizer.zero_grad()

        train_edge_index, train_edge_type = get_edge_info(data, "train")
        star_seed = data.star if hasattr(data, "star") else None

        if mode == "train":
            logits_node, logits_star, logits_lp = \
                model(data.x, train_edge_index, data.batch, star=star_seed, edge_type=train_edge_type)
        else:
            with torch.no_grad():
                logits_node, logits_star, logits_lp = \
                    model(data.x, train_edge_index, data.batch, star=star_seed, edge_type=train_edge_type)
        loss = None

        if node_classification:
            if mode == "train":
                mask = data.train_mask
            elif mode == "val":
                mask = data.val_mask
            elif mode == "test":
                mask = data.test_mask
            loss_ = model.nc_loss(logits_node[mask], data.y[mask], node_multi_label)
            loss = loss_ if loss is None else loss + loss_
            node_acc_count += model.nc_test(logits_node[mask],
                                            data.y[mask], node_multi_label)
            total_node += len(logits_node[mask])
        if graph_classification:
            loss_ = model.gc_loss(logits_star, data.y, graph_multi_label)
            loss = loss_ if loss is None else loss + loss_
            graph_acc_count += model.gc_test(logits_star, data.y, node_multi_label)
            total_graph += len(logits_star)
        if link_prediction:
            pei, pet = get_edge_info(data, mode)
            if mode == "train":
                if train_neg_sampling_queue is None:
                    train_neg_sampling_queue = Queue(maxsize=30)
                    train_true_tuples = torch.stack([pei[0], pet, pei[1]], dim=0).t().cpu().numpy()
                    train_true_tuples = set([tuple(l) for l in train_true_tuples.tolist()])
                    build_neg_sampling(pei.cpu(), pet.cpu(), train_true_tuples, logits_lp.size(0), 1,
                                       train_neg_sampling_queue, 10)
                if train_neg_sampling_queue.empty():
                    print("train neg sampling queue is empty,waiting...")
                nei, net = train_neg_sampling_queue.get()
            else:
                if test_neg_sampling_queue is None:
                    test_neg_sampling_queue = Queue(maxsize=30)
                    val_neg_sampling_queue = Queue(maxsize=30)
                    test_true_tuples = torch.stack([data.edge_index[0], data.edge_type, data.edge_index[1]],
                                                   dim=0).t().cpu().numpy()
                    test_true_tuples = set([tuple(l) for l in test_true_tuples.tolist()])
                    # build_neg_sampling(pei.cpu(), pet.cpu(), test_true_tuples, logits_lp.size(0), 1,
                    #                    test_neg_sampling_queue, 5)
                    # build_neg_sampling(pei.cpu(), pet.cpu(), test_true_tuples, logits_lp.size(0), 1,
                    #                    val_neg_sampling_queue, 5)
                # if test_neg_sampling_queue.empty():
                #     print("test neg sampling queue is empty,waiting...")
                if mode == "val":
                    nei, net = data.val_neg_edge_index, data.val_neg_edge_index.new_zeros(
                        (data.val_neg_edge_index.size(-1),))
                    # nei, net = val_neg_sampling_queue.get()
                elif mode == "test":
                    nei, net = data.test_neg_edge_index, data.test_neg_edge_index.new_zeros(
                        (data.test_neg_edge_index.size(-1),))
                    # nei, net = test_neg_sampling_queue.get()

            nei, net = nei.to(pei.device), net.to(pei.device)
            ei = torch.cat([pei, nei], dim=-1)
            et = torch.cat([pet, net], dim=-1)

            pred = model.lp_score(logits_lp, ei, et)
            y = torch.cat([logits_lp.new_ones(pei.size(-1)), logits_lp.new_zeros(nei.size(-1))], dim=0)

            loss_ = model.lp_loss(pred, y)
            loss = loss_ if loss is None else loss + loss_
            lp_auc, lp_ap = model.lp_test(pred, y)
            if not mode == "train" and cal_mrr_score:
                model.lp_log(logits_lp, pei, pet, data.edge_index, data.edge_type)

        total_loss += loss.item() * num_graphs
        if mode == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
    node_acc = float(node_acc_count) / total_node if node_classification else -1
    graph_acc = float(graph_acc_count) / total_graph if graph_classification else -1
    return total_loss / data_count, node_acc, graph_acc, lp_auc, lp_ap


def train_inductive(model, optimizer, loader, device, node_classification, node_multi_label, graph_classification,
                    graph_multi_label, link_prediction, mode="train", cal_mrr_score=False):
    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_node = 0
    node_acc_count = 0
    total_graph = 0
    graph_acc_count = 0
    lp_auc = lp_ap = None
    data_count = 0
    for data in loader:
        data_count += data.num_graphs
        num_graphs = data.num_graphs
        data = data.to(device)
        optimizer.zero_grad()
        star_seed = data.star if hasattr(data, "star") else None
        if mode == "train":
            logits_node, logits_star, logits_lp = \
                model(data.x, data.edge_index, data.batch, star=star_seed,
                      edge_type=data.edge_type if hasattr(data, "edge_type") else None)
        else:
            with torch.no_grad():
                logits_node, logits_star, logits_lp = \
                    model(data.x, data.edge_index, data.batch, star=star_seed,
                          edge_type=data.edge_type if hasattr(data, "edge_type") else None)

        loss = None
        if node_classification:
            loss_ = model.nc_loss(logits_node, data.y, node_multi_label)
            loss = loss_ if loss is None else loss + loss_
            node_acc_count += model.nc_test(logits_node, data.y, node_multi_label)
            total_node += len(logits_node)
        if graph_classification:
            loss_ = model.gc_loss(logits_star, data.y, graph_multi_label)
            loss = loss_ if loss is None else loss + loss_
            graph_acc_count += model.gc_test(logits_star, data.y, node_multi_label)
            total_graph += len(logits_star)

        total_loss += loss.item() * num_graphs
        if mode == "train":
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
            optimizer.step()
    node_acc = float(node_acc_count) / total_node if node_classification else -1
    graph_acc = float(graph_acc_count) / total_graph if graph_classification else -1
    return total_loss / data_count, node_acc, graph_acc, lp_auc, lp_ap


def trainer(args, DATASET, train_loader, val_loader, test_loader, transductive=False,
            num_features=0, num_node_class=0, num_graph_class=0, test_per_epoch=1, val_per_epoch=1, max_epoch=2000,
            save_per_epoch=100, load_model=False, cal_mrr_score=False,
            node_multi_label=False, graph_multi_label=False, link_prediction=False):
    if transductive:
        train = train_transductive
    else:
        train = train_inductive
        assert (not link_prediction), "link prediction only works in transductive mode"

    print(args)
    print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Print out args @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    tab_printer(args)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    node_classification = num_node_class > 0
    graph_classification = num_graph_class > 0
    model = GraphStar(num_features=num_features, num_node_class=num_node_class,
                      num_graph_class=num_graph_class, hid=args.hidden, num_star=args.num_star,
                      star_init_method=args.star_init_method, link_prediction=link_prediction,
                      heads=args.heads, cross_star=args.cross_star, num_layers=args.num_layers,
                      cross_layer=args.cross_layer, dropout=args.dropout, coef_dropout=args.coef_dropout,
                      residual=args.residual,
                      residual_star=args.residual_star, layer_norm=args.layer_norm, activation=args.activation,
                      layer_norm_star=args.layer_norm_star, use_e=args.use_e, num_relations=args.num_relations,
                      one_hot_node=args.one_hot_node, one_hot_node_num=args.one_hot_node_num,
                      relation_score_function=args.relation_score_function,
                      additional_self_loop_relation_type=args.additional_self_loop_relation_type,
                      additional_node_to_star_relation_type=args.additional_node_to_star_relation_type)

    model.to(args.device)

    if DATASET in ['MR_win10_no_prefeat_no_repeat', 'MR_win10_no_prefeat_repeat', 'R8_win10_no_prefeat_no_repeat',
                   'R8_win10_no_prefeat_repeat', '20ng_win10_no_prefeat_repeat', 'R52_win10_no_prefeat_no_repeat',
                   'R52_win10_no_prefeat_repeat']:
        print('process text classification')

    print(time.asctime(time.localtime(time.time())))

    lr = args.lr
    tw.init_writer(DATASET)

    #Create directory, if it doesn't already exists
    PATH = 'output'
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=args.patience, verbose=True, factor=0.5, cooldown=30,
                                  min_lr=lr / 100)
    max_node_acc = 0
    max_graph_acc = 0
    max_lp_auc = 0
    max_lp_ap = 0
    max_val_lp = 0
    gc_accs = []
    max_gc_epoch_idx = 0

    for epoch in range(0, max_epoch + 1):
        train_loss, train_node_acc, train_graph_acc, train_lp_auc, train_lp_ap = \
            train(model, optimizer, train_loader,
                  args.device,
                  node_classification,
                  node_multi_label,
                  graph_classification,
                  graph_multi_label,
                  link_prediction, mode="train")
        if epoch % val_per_epoch == 0:
            val_loss, val_node_acc, val_graph_acc, val_lp_auc, val_lp_ap = \
                train(model, optimizer, val_loader,
                      args.device,
                      node_classification,
                      node_multi_label,
                      graph_classification,
                      graph_multi_label,
                      link_prediction, mode="val", cal_mrr_score=cal_mrr_score)
        else:
            val_loss, val_node_acc, val_graph_acc, val_lp_auc, val_lp_ap = 0, 0, 0, 0, 0
        if epoch % test_per_epoch == 0:
            test_loss, test_node_acc, test_graph_acc, test_lp_auc, test_lp_ap = \
                train(model, optimizer, test_loader,
                      args.device,
                      node_classification,
                      node_multi_label,
                      graph_classification,
                      graph_multi_label,
                      link_prediction, mode="test", cal_mrr_score=cal_mrr_score)
        else:
            test_loss, test_node_acc, test_graph_acc, test_lp_auc, test_lp_ap = 0, 0, 0, 0, 0

        train_str = ""
        val_str = ""
        test_str = ""
        max_str = ""
        if node_classification:
            tw.writer.add_scalar('train/node_acc', train_node_acc, tw.train_steps)
            tw.writer.add_scalar('val/node_acc', val_node_acc, tw.val_steps)
            # tw.writer.add_scalar('test/node_acc', test_node_acc, tw.test_steps)
            max_node_acc = max(test_node_acc, max_node_acc)

            train_str += "NC Acc: {:.4f}, ".format(train_node_acc)
            val_str += "NC Acc: {:.4f}, ".format(val_node_acc)
            test_str += "NC Acc: {:.4f}, ".format(test_node_acc)
            max_str += "NC Acc: {:.4f}, ".format(max_node_acc)
        if graph_classification:
            tw.writer.add_scalar('train/graph_acc', train_graph_acc, tw.train_steps)

            if (test_graph_acc - max_graph_acc) > 1e-4:
                max_gc_epoch_idx = epoch

            tw.writer.add_scalar('val/graph_acc', val_graph_acc, tw.val_steps)
            # tw.writer.add_scalar('test/graph_acc', test_graph_acc, tw.test_steps)
            max_graph_acc = max(test_graph_acc, max_graph_acc)
            gc_accs.append(test_graph_acc)

            train_str += "GC Acc: {:.4f}, ".format(train_graph_acc)
            val_str += "GC Acc: {:.4f}, ".format(val_graph_acc)
            test_str += "GC Acc: {:.4f}, ".format(test_graph_acc)
            max_str += "GC Acc: {:.4f}, MaxE: {}, Hold: {}".format(max_graph_acc, max_gc_epoch_idx,
                                                                   epoch - max_gc_epoch_idx)
        if link_prediction:
            tw.writer.add_scalar('train/lp_auc', train_lp_auc, tw.train_steps)
            tw.writer.add_scalar('val/lp_auc', val_lp_auc, tw.val_steps)
            # tw.writer.add_scalar('test/lp_auc', test_lp_auc, tw.test_steps)
            tw.writer.add_scalar('train/lp_ap', train_lp_ap, tw.train_steps)
            tw.writer.add_scalar('val/lp_ap', val_lp_ap, tw.val_steps)
            # tw.writer.add_scalar('test/lp_ap', test_lp_ap, tw.test_steps)
            max_lp_auc = max(test_lp_auc, max_lp_auc)
            max_lp_ap = max(test_lp_ap, max_lp_ap)
            max_val_lp = max((val_lp_ap + val_lp_auc) / 2, max_val_lp)

            train_str += "LP AVG: {:.4f}, ".format(sum([train_lp_auc, train_lp_ap]) / 2)
            val_str += "LP AVG: {:.4f}, ".format(sum([val_lp_auc, val_lp_ap]) / 2)
            test_str += "LP AVG: {:.4f}, ".format(sum([test_lp_auc, test_lp_ap]) / 2)
            max_str += "LP AVG: {:.4f},VAL: {:.4f} ".format(sum([max_lp_auc, max_lp_ap]) / 2, max_val_lp)
        tw.writer.add_scalar('train/loss', train_loss, tw.train_steps)
        tw.writer.add_scalar('val/loss', val_loss, tw.val_steps)
        # tw.writer.add_scalar('test/loss', test_loss, tw.test_steps)

        log_str = 'Epoch: {:02d}, TRAIN Loss: {:.4f}, {} || VAL Loss: {:.4f}, {} || TEST Loss: {:.4f}, {} || Max {}'.format(
            epoch, train_loss, train_str, val_loss, val_str, test_loss, test_str, max_str)
        print("\033[1;32m", DATASET, "\033[0m", log_str)
        # print("use time : %f" % (time.time()-start))
        if epoch % save_per_epoch == 0:
            torch.save(model, os.path.join("output", DATASET + ".pkl"))
        scheduler.step(train_loss)
    tw.writer.close()
    return max_graph_acc, gc_accs

def negative_sampling(pei, pet, true_triples, num_nodes, count=1):
    res_nei = []
    res_net = []

    for i in range(pei.size(-1)):
        head, rel, tail = pei[0][i], pet[i], pei[1][i]

        false_head = []
        false_tail = []
        while len(false_head) < count:
            negative_sample = np.random.randint(num_nodes, size=max(count * 2, 0))
            negative_sample = [x for x in negative_sample if (x, rel, tail) not in true_triples]
            false_head.extend(negative_sample)
        while len(false_tail) < count:
            negative_sample = np.random.randint(num_nodes, size=max(count * 2, 0))
            negative_sample = [x for x in negative_sample if (head, rel, x) not in true_triples]
            false_tail.extend(negative_sample)

        nei_0 = torch.cat([pei.new_full((count,), head), pei.new_tensor(false_head[:count])], dim=0)
        nei_1 = torch.cat([pei.new_tensor(false_tail[:count]), pei.new_full((count,), tail)], dim=0)
        nei = torch.stack([nei_0, nei_1], dim=0)
        net = pei.new_full((count * 2,), rel)
        res_nei.append(nei)
        res_net.append(net)
    return torch.cat(res_nei, dim=-1), torch.cat(res_net, dim=-1)


def loop_negative_sampling(pei, pet, true_tuples, num_node, count, queue):
    while True:
        nei, net = negative_sampling(pei, pet, true_tuples, num_node, count)
        queue.put((nei, net))


def build_neg_sampling(pei, pet, true_tuples, num_node, count, queue, num_thread):
    for i in range(num_thread):
        p = Process(target=loop_negative_sampling, args=(pei, pet, true_tuples, num_node, count, queue), daemon=True)
        p.start()

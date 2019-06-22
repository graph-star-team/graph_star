import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool as gap
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics

# from .graph_star_conv import GraphStarConv
# from .graph_star_conv_multi_rel import GraphStarConv
from .graph_star_conv_multi_rel_super_attn import GraphStarConv
from .star_attn import StarAttn
from .cross_layer_attn import CrossLayerAttn
import sys

sys.path.append("..")
import utils.tensorboard_writer as tw

EPS = 1e-15


class GraphStar(nn.Module):
    def __init__(self, num_features, num_node_class, num_graph_class, hid, num_star=4, cross_star=False, heads=6,
                 num_relations=18, one_hot_node=True, one_hot_node_num=0, star_init_method="attn",
                 link_prediction=False, coef_dropout=0.2,
                 dropout=0.1, residual=True, residual_star=True, layer_norm=True, layer_norm_star=True, use_e=True,
                 num_layers=3, cross_layer=False, activation=None, additional_self_loop_relation_type=False,
                 additional_node_to_star_relation_type=False, relation_score_function="DistMult"):
        super(GraphStar, self).__init__()
        self.one_hot_node = one_hot_node
        self.num_graph_class = num_graph_class
        self.cross_layer = cross_layer
        self.num_layers = num_layers
        self.use_e = use_e
        self.num_star = num_star
        self.num_features = num_features
        self.hid = hid
        self.node_classification = num_node_class > 0
        self.graph_classification = num_graph_class > 0
        self.link_prediction = link_prediction
        self.dropout = dropout
        self.coef_dropout = coef_dropout
        assert star_init_method in ["mean", "attn"], "star init method must be mean or attn"
        self.star_init_method = star_init_method

        self.gamma = nn.Parameter(
            torch.Tensor([24.0]),
            requires_grad=False
        )

        assert callable(getattr(self, relation_score_function)), "relation score function not found."
        self.relation_score_function = getattr(self, relation_score_function)
        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hid]),
            requires_grad=False
        )
        if relation_score_function == "pRotatE":
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        if num_relations > 1:
            assert additional_self_loop_relation_type and additional_node_to_star_relation_type

        if additional_self_loop_relation_type:
            self.self_loop_relation_type = num_relations
            num_relations += 1
        else:
            self.self_loop_relation_type = 0

        if additional_node_to_star_relation_type:
            self.node_to_star_relation_type = num_relations
            num_relations += 1
        else:
            self.node_to_star_relation_type = 0
        self.num_relations = num_relations

        if one_hot_node:
            assert one_hot_node_num > 0
            self.x_embedding = torch.nn.Embedding(one_hot_node_num, hid)
        else:
            self.fl = nn.Linear(num_features, hid)
        self.star_init = StarAttn(heads=self.num_star, use_star=False, cross_star=False,
                                  in_channels=hid, out_channels=hid * self.num_star,
                                  dropout=dropout, coef_dropout=coef_dropout, residual=False,
                                  layer_norm=layer_norm_star, activation=activation)

        self.conv_list = nn.ModuleList()
        self.star_attn_list = nn.ModuleList()
        for i in range(num_layers):
            # update node
            conv = GraphStarConv(hid, hid, heads=heads, num_star=num_star, dropout=dropout,
                                 coef_dropout=coef_dropout, layer=i,
                                 residual=residual, layer_norm=layer_norm, use_e=use_e, activation=activation,
                                 num_relations=num_relations, self_loop_relation_type=self.self_loop_relation_type,
                                 node_to_star_relation_type=self.node_to_star_relation_type)
            # update star
            star_attn = StarAttn(heads=heads, use_star=True, cross_star=cross_star, in_channels=hid, layer=i,
                                 out_channels=hid, dropout=dropout, coef_dropout=coef_dropout, residual=residual_star,
                                 layer_norm=layer_norm_star, activation=activation)

            self.conv_list.append(conv)
            self.star_attn_list.append(star_attn)

        if self.node_classification:
            self.node_linear = nn.Linear(hid, num_node_class)
        if self.graph_classification:
            self.star_linear = nn.Linear(hid, num_graph_class)

        self.gcl1 = nn.Linear(hid * 2, hid)
        self.gcl2 = nn.Linear(hid, hid // 2)
        self.gcl3 = nn.Linear(hid // 2, num_graph_class)

        if cross_layer:
            self.cross_layer_attn = CrossLayerAttn(heads=heads, use_star=False, cross_star=False, in_channels=hid,
                                                   out_channels=hid, dropout=dropout, coef_dropout=coef_dropout,
                                                   residual=False, layer_norm=layer_norm_star)
        self.rl = nn.Linear(num_relations, hid)
        self.RW = Parameter(torch.empty(num_relations, hid).uniform_(-0.1, 0.1))
        self.LP_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index, batch, star=None, y=None, edge_type=None):
        # times = []
        if self.training:
            tw.train_steps += 1
        else:
            tw.val_steps += 1
        num_node = x.size(0)
        num_graph = len(torch.bincount(batch))
        _edge_index = edge_index

        if self.one_hot_node:
            x = self.x_embedding(x)
        else:
            stars = None
            x = self.fl(x)
            x = F.relu(x)

        if edge_type is None:
            edge_type = edge_index.new_zeros((edge_index.size(-1)))
        edge_index, edge_type = self.add_self_loop_edge(edge_index, edge_type, num_node)

        if self.num_star > 0:
            edge_index, edge_type = self.add_star_edge(edge_index, edge_type, num_node, batch)
            # times.append(time.time())
            if star is None:
                star_seed = gap(x, batch)
            else:
                star = self.fl(star)
                star_seed = F.relu(star)
                # star_seed = star
            star_seed = star_seed.unsqueeze(1)
            # times.append(time.time())

            if self.star_init_method == "attn":
                stars = self.star_init(star_seed, x, batch).view(num_graph, self.num_star, self.hid)
                # times.append(time.time())
            elif self.star_init_method == "mean":
                assert self.num_star == 1
                stars = torch.unsqueeze(star_seed, 1)

        x_list = []
        stars_list = []
        # times.append(time.time())

        for conv, star_attn, i in zip(self.conv_list, self.star_attn_list, range(len(self.conv_list))):
            # update node
            x = conv(x, stars.view(-1, self.hid), None, edge_index, edge_type=edge_type)
            # update star
            stars = star_attn(stars, x, batch)

            if self.cross_layer:
                x_list.append(x[:num_node])
                stars_list.append(stars)
        # if y is not None:
        #     meta = torch.cat([y, torch.full((1,self.num_star), 2, dtype=y.dtype, device=y.device).view(-1)], dim=0)
        #     tw.writer.add_embedding(torch.cat([x, stars], dim=0), metadata=meta,
        #                             global_step=(tw.train_steps if self.training else tw.val_steps),
        #                             tag=("train/" if self.training else "eval/") + "nodes")
        # times.append(time.time())

        if self.cross_layer:
            if self.node_classification or self.link_prediction:
                target_list = x_list
            elif self.graph_classification:
                target_list = stars_list
            if self.node_classification or self.graph_classification or self.link_prediction:
                layer_res = torch.cat(target_list, dim=0).view(self.num_layers, -1, self.hid)  # num_layer,num_node,hid
                layer_res = layer_res.transpose(1, 0)  # num_node,num_layer,hid
                mean_res = torch.mean(layer_res, dim=1, keepdim=True)  # num_node,1,hid
                res = self.cross_layer_attn(mean_res, layer_res)
            if self.node_classification or self.link_prediction:
                x = res
            elif self.graph_classification:
                stars = res
        x_lp = x
        x = self.node_linear(x) if self.node_classification else x
        stars = self.star_linear(stars) if self.graph_classification else stars
        if self.graph_classification:
            if len(stars.shape) == 3:
                stars = stars.mean(dim=1)
        # times.append(time.time())
        # times = [t - times[i - 1] for i, t in enumerate(times) if i > 0]

        return x, stars, x_lp

    def nc_loss(self, x, y, multi_label=False):
        if multi_label:
            node_loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            node_loss_op = torch.nn.CrossEntropyLoss()

        return node_loss_op(x, y)

    def nc_test(self, x, y, multi_label=False):
        if multi_label:
            micro_f1 = metrics.f1_score(y.cpu().detach().numpy(), (x > 0).cpu().detach().numpy(), average='micro')
            node_acc_count = micro_f1 * len(x)
        else:
            y = y.cpu()
            pred = torch.argmax(F.softmax(x, dim=1), dim=1).cpu()
            node_acc_count = metrics.accuracy_score(y,
                                                    pred,
                                                    normalize=False)

        return node_acc_count

    def gc_loss(self, x, y, multi_label=False):
        if multi_label:
            graph_loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            graph_loss_op = torch.nn.CrossEntropyLoss()

        return graph_loss_op(x, y)

    def gc_test(self, x, y, multi_label=False):
        graph_acc_count = metrics.accuracy_score(y.cpu(),
                                                 torch.argmax(F.softmax(x, dim=1), dim=1).cpu(),
                                                 normalize=False)

        return graph_acc_count

    def lp_score(self, z, edge_index, edge_type):
        z = F.dropout(z, 0.5, training=self.training)
        pred = self.relation_score_function(z[edge_index[0]].unsqueeze(1),
                                            self.RW[edge_type].unsqueeze(1),
                                            z[edge_index[1]].unsqueeze(1)
                                            )

        return pred

    def lp_log(self, z, pos_edge_index, pos_edge_type, known_edge_index, known_edge_type):
        dt, dev = pos_edge_index.dtype, pos_edge_index.device
        ranks = []

        # head batch
        for i in range(len(pos_edge_type)):
            pei = torch.stack([torch.arange(0, z.size(0), dtype=dt, device=dev),
                               torch.full((z.size(0),), pos_edge_index[1][i], dtype=dt, device=dev)], dim=0)
            pet = torch.full((z.size(0),), pos_edge_type[i], dtype=dt, device=dev)
            pred = self.lp_score(z, pei, pet)
            rank = (pred >= pred[pos_edge_index[0][i]]).sum().item()

            # filter
            filter_idx = ((known_edge_index[1] == pos_edge_index[1][i]) * (
                    known_edge_type == pos_edge_type[i])).nonzero().view(-1)
            filter_idx = known_edge_index[0][filter_idx]
            rank -= (pred[filter_idx] >= pred[pos_edge_index[0][i]]).sum().item()
            rank += 1
            ranks.append(rank)

        # tail batch
        for i in range(len(pos_edge_type)):
            pei = torch.stack([torch.full((z.size(0),), pos_edge_index[0][i], dtype=dt, device=dev),
                               torch.arange(0, z.size(0), dtype=dt, device=dev)], dim=0)
            pet = torch.full((z.size(0),), pos_edge_type[i], dtype=dt, device=dev)
            pred = self.lp_score(z, pei, pet)

            rank = (pred >= pred[pos_edge_index[1][i]]).sum().item()

            # filter
            filter_idx = ((known_edge_index[0] == pos_edge_index[0][i]) * (
                    known_edge_type == pos_edge_type[i])).nonzero().view(-1)
            filter_idx = known_edge_index[1][filter_idx]
            rank -= (pred[filter_idx] >= pred[pos_edge_index[1][i]]).sum().item()
            rank += 1
            ranks.append(rank)
        ranks = np.array(ranks)

        print("MRR: %f, MR: %f, HIT@1: %f, HIT@3: %f, HIT@10: %f" % (
            (1 / ranks).sum() / len(ranks),
            (ranks).sum() / len(ranks),
            (ranks <= 1).sum() / len(ranks),
            (ranks <= 3).sum() / len(ranks),
            (ranks <= 10).sum() / len(ranks)
        ))

    def lp_loss(self, pred, y):
        return self.LP_loss(pred.squeeze(1), y)

    def lp_test(self, pred, y):
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def add_star_edge(self, edge_index, edge_type, x_size, batch):
        dtype, device = edge_index.dtype, edge_index.device

        row = torch.arange(start=0, end=x_size, dtype=dtype, device=device)
        b1 = batch * self.num_star + x_size
        for i in range(self.num_star):
            col = b1 + i
            edge_index = torch.cat([edge_index, torch.stack([row, col], dim=0)], dim=1)
            edge_type = torch.cat([edge_type, edge_type.new_full((x_size,), self.node_to_star_relation_type)])

        return edge_index, edge_type

    def add_self_loop_edge(self, edge_index, edge_type, x_size):
        dtype, device = edge_index.dtype, edge_index.device
        tmp = torch.arange(0, x_size, dtype=dtype, device=device)
        tmp = torch.stack([tmp, tmp], dim=0)
        edge_index = torch.cat([edge_index, tmp], dim=1)

        edge_type = torch.cat([edge_type, edge_type.new_full((x_size,), self.self_loop_relation_type)], dim=0)
        return edge_index, edge_type

    def DistMult(self, head, relation, tail):
        score = head * relation * tail

        return score.sum(dim=2)

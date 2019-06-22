import torch
import torch.nn as nn

import torch.nn.functional as F
import math
import sys
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.utils import scatter_

sys.path.append("..")
import utils.tensorboard_writer as tw


class StarAttn(nn.Module):

    def __init__(self, heads, use_star, cross_star, in_channels, out_channels, dropout=0.1, residual=True, layer=0,
                 coef_dropout=0.2, layer_norm=True, activation=F.elu):
        super(StarAttn, self).__init__()
        self.coef_dropout = coef_dropout
        self.layer = layer
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.residual = residual
        self.heads = heads
        self.use_star = use_star
        self.cross_star = cross_star
        # self.num_star = num_star

        self.dropout = dropout
        self.layer_norm = layer_norm
        self.activation = activation

        self.Wq = torch.nn.Linear(in_channels, out_channels)
        self.Wk = torch.nn.Linear(in_channels, out_channels)
        self.Wv = torch.nn.Linear(in_channels, out_channels)
        assert out_channels % heads == 0
        self.size_pre_head = int(self.out_channels / self.heads)

        # self.sWo = torch.nn.Linear(out_channels, out_channels)
        self.sLayerNorm = torch.nn.LayerNorm(out_channels)

    def forward(self, stars, nodes, batch):
        dtype, device = batch.dtype, batch.device
        num_star = stars.size(1)
        b = num_star * batch + len(nodes)

        edge_index = batch.new_empty((2, 0))
        col = torch.arange(start=0, end=len(nodes), dtype=dtype, device=device)

        # add star to node edge
        for i in range(num_star):
            row = b + i
            edge_index = torch.cat([edge_index, torch.stack([row, col], dim=0)], dim=1)

        # add star self loop
        if self.use_star:
            star_row = torch.arange(start=len(nodes), end=len(nodes) + len(stars), dtype=dtype, device=device)
            edge_index = torch.cat([edge_index, torch.stack([star_row, star_row], dim=0)], dim=1)
        # TODO add cross star!

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x = torch.cat([nodes, stars.view(-1, self.in_channels)], dim=0)

        xq = self.Wq(x).view(-1, self.heads, self.size_pre_head)
        xk = self.Wk(x).view(-1, self.heads, self.size_pre_head)
        xv = self.Wv(x).view(-1, self.heads, self.size_pre_head)

        xq = torch.index_select(xq, 0, edge_index_i)
        xk = torch.index_select(xk, 0, edge_index_j)
        xv = torch.index_select(xv, 0, edge_index_j)

        score = self.cal_att_score(xq, xk, self.heads)
        coef = softmax(score, edge_index_i, len(x))

        # TODO add tensorboard
        # [:-num_star]  is star to node
        # [-num_star:] is star self loop

        coef = F.dropout(coef, p=self.coef_dropout, training=self.training)
        xv = F.dropout(xv, p=self.dropout, training=self.training)

        out = xv * coef.view(-1, self.heads, 1)

        out = scatter_("add", out, edge_index_i)[len(nodes):]
        new_stars = out.view(-1, num_star, self.out_channels)

        if self.activation is not None:
            new_stars = self.activation(new_stars)
        if self.residual:
            new_stars = new_stars + stars
        if self.layer_norm:
            new_stars = self.sLayerNorm(new_stars)
        return new_stars

    def cal_att_score(self, q, k, heads):
        out_channel = q.size(-1)
        score = torch.matmul(q.view(-1, heads, 1, out_channel), k.view(-1, heads, out_channel, 1)).view(
            -1, heads)
        score = score / math.sqrt(out_channel)
        return score

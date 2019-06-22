import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import matplotlib
import math
import sys

sys.path.append("..")
import utils.tensorboard_writer as tw

class CrossLayerAttn(torch.nn.Module):

    def __init__(self, heads, use_star, cross_star, in_channels, out_channels, dropout=0.1, coef_dropout=0.1,
                 residual=True, layer_norm=True, activation=torch.nn.ReLU()):
        super(CrossLayerAttn, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.residual = residual
        self.heads = heads
        self.use_star = use_star
        self.cross_star = cross_star

        self.dropout = dropout
        self.coef_dropout = coef_dropout
        self.layer_norm = layer_norm
        self.activation = activation

        self.sWq = torch.nn.Linear(in_channels, out_channels)
        self.sWk = torch.nn.Linear(in_channels, out_channels)
        self.sWv = torch.nn.Linear(in_channels, out_channels)
        assert out_channels % heads == 0
        self.channel_per_head = int(self.out_channels / self.heads)

        self.sWo = torch.nn.Linear(out_channels, out_channels)
        self.sLayerNorm = torch.nn.LayerNorm(out_channels)

    def forward(self, q, kv):
        num_node = kv.size(0)
        num_layer = kv.size(1)

        k = self.sWk(kv).view(num_node, num_layer, self.heads,
                              self.channel_per_head, 1)  # num_node,num_layer,heads,channel_per_head,1
        v = self.sWv(kv).view(num_node, num_layer, self.heads,
                              self.channel_per_head)  # num_node,num_layer,heads,channel_per_head

        q = self.sWq(q)  # num_node,1,out_channels
        q = q.squeeze(1)  # num_node,out_channels
        q = q.view(-1, 1, self.heads, 1, self.channel_per_head)  # num_node,heads,1,channel_per_head,1

        score = torch.matmul(q, k).view(num_node, num_layer, self.heads, 1)  # num_node,num_layer,heads,1
        score = score / math.sqrt(self.channel_per_head)  # num_node,num_layer,heads,1

        # score = F.leaky_relu(score)
        coef = F.softmax(score, dim=1)
        coef = F.dropout(coef, self.coef_dropout, training=self.training)  # num_node,num_layer,heads,1

        # show = score.mean(dim=0).mean(dim=1).view(num_layer).cpu().detach().numpy()
        # name = "train/" if self.training else "eval/"
        # fig = plt.figure()
        # plt.bar(range(len(show)), show)
        # tw.writer.add_figure(name + "cross_layer_attn_coef", fig, tw.train_steps if self.training else tw.val_steps)

        res = (coef * v).sum(dim=1).view(num_node, self.out_channels)  # num_node,out_channels

        return res

    def cal_att_score(self, q, k, heads):
        out_channel = q.size(-1)
        score = torch.matmul(q.view(-1, heads, 1, out_channel), k.view(-1, heads, out_channel, 1)).view(
            -1, heads)
        score = score / math.sqrt(q.size(-1))
        return score

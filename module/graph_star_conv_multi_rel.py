import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math


class GraphStarConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{j} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        layer=0,
        heads=1,
        dropout=0.1,
        num_star=1,
        residual=True,
        layer_norm=True,
        use_e=True,
        activation=None,
        num_relations=1,
    ):
        super(GraphStarConv, self).__init__()
        self.layer = layer
        self.use_e = use_e
        self.layer_norm = layer_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.num_star = num_star
        self.residual = residual
        self.activation = activation

        assert out_channels % heads == 0
        self.size_pre_head = out_channels // heads
        self.nWq = torch.nn.ModuleList(
            [torch.nn.Linear(in_channels, out_channels) for _ in range(num_relations)]
        )
        self.nWk = torch.nn.ModuleList(
            [torch.nn.Linear(in_channels, out_channels) for _ in range(num_relations)]
        )
        self.nWv = torch.nn.ModuleList(
            [torch.nn.Linear(in_channels, out_channels) for _ in range(num_relations)]
        )
        self.nWo = torch.nn.ModuleList(
            [torch.nn.Linear(out_channels, out_channels) for _ in range(num_relations)]
        )

        self.nLayerNorm = torch.nn.LayerNorm(out_channels)
        self.num_relations = num_relations

    def forward(self, x, stars, init_x, edge_index, edge_type=None):
        """"""
        num_node = x.size(0)
        nodes = x

        if init_x is not None:
            nodes = torch.cat([nodes, init_x], dim=0)

        if stars is not None:
            nodes = torch.cat([nodes, stars], dim=0)

        nodes = self.propagate(
            "add", edge_index, edge_type, x=nodes, num_nodes=num_node
        )

        return nodes

    def propagate(self, aggr, edge_index, edge_type, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            aggr (string): Takes in an aggregation scheme (:obj:`"add"`,
                :obj:`"mean"` or :obj:`"max"`).
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[M, N]` (can be directed or
                undirected).
            size (int, optional): The output node size :math:`M`. If
                :obj:`None`, the output node size will get automatically
                inferrred by assuming a symmetric adjacency matrix as input.
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """
        assert aggr in ["add", "mean", "max"]
        kwargs["edge_index"] = edge_index
        x = kwargs["x"]
        num_nodes = kwargs["num_nodes"]

        out_list = []
        for i in range(self.num_relations):
            edge_index_mask = edge_type == i
            if edge_index_mask.sum() > 0:

                edge_index_i = edge_index[0][edge_index_mask]
                edge_index_j = edge_index[1][edge_index_mask]

                # 同样的功能，只是根据稀疏程度决定运算顺序以提高性能
                if edge_index_mask.sum() < num_nodes:
                    xi = torch.index_select(x, 0, edge_index_i)
                    xj = torch.index_select(x, 0, edge_index_j)

                    xq = self.nWq[i](xi).view(-1, self.heads, self.size_pre_head)
                    xk = self.nWk[i](xj).view(-1, self.heads, self.size_pre_head)
                    xv = self.nWv[i](xj).view(-1, self.heads, self.size_pre_head)
                else:
                    xq = self.nWq[i](x).view(-1, self.heads, self.size_pre_head)
                    xk = self.nWk[i](x).view(-1, self.heads, self.size_pre_head)
                    xv = self.nWv[i](x).view(-1, self.heads, self.size_pre_head)

                    xq = torch.index_select(xq, 0, edge_index_i)
                    xk = torch.index_select(xk, 0, edge_index_j)
                    xv = torch.index_select(xv, 0, edge_index_j)

                out = self.message(xq, xk, xv, edge_index_i, num_nodes)

                if (aggr=="add"):
                    out = torch.scatter_add(out, edge_index_i, dim_size=size)
                elif (aggr=="mean"):
                    out = torch.scatter_mean(out, edge_index_i, dim_size=size)
                else:
                    out = torch.scatter_max(out, edge_index_i, dim_size=size)
                out = self.update(out)
                if out.size(0) < num_nodes:
                    out = torch.cat(
                        [out, out.new_zeros([num_nodes - out.size(0), out.size(-1)])]
                    )
                out = self.nWo[i](out)
            else:
                out = x.new_zeros([num_nodes, self.out_channels])
            out_list.append(out)
        out = torch.stack(out_list, dim=0)
        out = torch.mean(out, dim=0)

        if self.activation is not None:
            out = self.activation(out)
        if self.residual:
            out = out + x[:num_nodes]

        if self.layer_norm:
            out = self.nLayerNorm(out)
        return out

    def message(self, x_q, x_k, x_v, edge_index_i, num_nodes):
        score = self.cal_att_score(x_q, x_k, self.heads)
        # score = F.leaky_relu(score)
        score = softmax(score, edge_index_i, num_nodes)

        # score = F.dropout(score, p=self.dropout, training=self.training)
        x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        return x_v * score.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)

        return aggr_out

    def cal_att_score(self, q, k, heads):
        out_channel = q.size(-1)
        score = torch.matmul(
            q.view(-1, heads, 1, out_channel), k.view(-1, heads, out_channel, 1)
        ).view(-1, heads)
        score = score / math.sqrt(out_channel)
        return score

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from model.graph_transformer_layer import GraphTransformerLayer

class GraphConvLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout=0.0,
        layer_norm=True,
        batch_norm=False,
        residual=True
    ):
        super(GraphConvLayer, self).__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual

        self.conv = GraphConv(
            in_dim,
            out_dim,
            norm='both',
            weight=True,
            bias=True,
            allow_zero_in_degree=True
        )

        if residual:
            if in_dim != out_dim:
                self.residual_proj = nn.Linear(in_dim, out_dim)
            else:
                self.residual_proj = nn.Identity()

        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, g, h):
        h_in1 = h

        h = self.conv(g, h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h + self.residual_proj(h_in1)

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h

        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h + h_in2

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h


class GraphConv1(nn.Module):
    def __init__(self, device, n_layers, node_dim, hidden_dim, out_dim, n_heads, dropout):
        super(GraphConv1, self).__init__()

        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True

        self.linear_h = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphConvLayer(
                hidden_dim,
                hidden_dim,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual
            )
            for _ in range(n_layers - 1)
        ])

        self.layers.append(
            GraphConvLayer(
                hidden_dim,
                out_dim,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual
            )
        )

    def forward(self, g):
        g = g.to(self.device)

        with g.local_scope():
            h = g.ndata['meta_sim'].float().to(self.device)

            h = self.linear_h(h)

            for conv in self.layers:
                h = conv(g, h)

            return h




class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dim, hidden_dim, out_dim, n_heads, dropout):
        super(GraphTransformer, self).__init__()

        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        # self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))

    def forward(self, g):
        # input embedding
        g = g.to(self.device)
        with g.local_scope():
            h = g.ndata['meta_sim'].float().to(self.device)

            h = self.linear_h(h)
            # h = self.in_feat_dropout(h)

            # convnets
            for conv in self.layers:
                h= conv(g, h)

            # h = dgl.mean_nodes(g, 'h')

            return h
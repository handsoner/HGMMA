import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv


class HeteroGraphConvLayer(nn.Module):
    def __init__(
        self,
        node_types,
        in_dim,
        out_dim,
        dropout=0.0,
        layer_norm=True,
        batch_norm=False,
        residual=True
    ):
        super(HeteroGraphConvLayer, self).__init__()

        self.node_types = node_types
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual

        self.self_linear = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim)
            for ntype in node_types
        })

        self.rel_conv = nn.ModuleDict({
            'metabolite__association__microbe': GraphConv(
                in_dim,
                out_dim,
                norm='both',
                weight=True,
                bias=True,
                allow_zero_in_degree=True
            ),
            'microbe__rev_association__metabolite': GraphConv(
                in_dim,
                out_dim,
                norm='both',
                weight=True,
                bias=True,
                allow_zero_in_degree=True
            )
        })

        if residual:
            self.residual_proj = nn.ModuleDict({
                ntype: nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
                for ntype in node_types
            })

        if layer_norm:
            self.layer_norm1 = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim)
                for ntype in node_types
            })
            self.layer_norm2 = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim)
                for ntype in node_types
            })

        if batch_norm:
            self.batch_norm1 = nn.ModuleDict({
                ntype: nn.BatchNorm1d(out_dim)
                for ntype in node_types
            })
            self.batch_norm2 = nn.ModuleDict({
                ntype: nn.BatchNorm1d(out_dim)
                for ntype in node_types
            })

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

    def _rel_key(self, canonical_etype):
        src_type, edge_type, dst_type = canonical_etype
        return f'{src_type}__{edge_type}__{dst_type}'

    def forward(self, g, h_dict):
        out_dict = {
            ntype: self.self_linear[ntype](h_dict[ntype])
            for ntype in self.node_types
        }

        for canonical_etype in g.canonical_etypes:
            src_type, edge_type, dst_type = canonical_etype
            rel_key = self._rel_key(canonical_etype)

            if rel_key not in self.rel_conv:
                continue

            rel_graph = g[canonical_etype]
            src_feat = h_dict[src_type]
            dst_feat = h_dict[dst_type]

            msg = self.rel_conv[rel_key](rel_graph, (src_feat, dst_feat))
            out_dict[dst_type] = out_dict[dst_type] + msg

        new_h_dict = {}

        for ntype in self.node_types:
            h = out_dict[ntype]
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

            if self.residual:
                h = h + self.residual_proj[ntype](h_dict[ntype])

            if self.layer_norm:
                h = self.layer_norm1[ntype](h)

            if self.batch_norm:
                h = self.batch_norm1[ntype](h)

            h_in2 = h

            h = self.FFN_layer1(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.FFN_layer2(h)

            if self.residual:
                h = h + h_in2

            if self.layer_norm:
                h = self.layer_norm2[ntype](h)

            if self.batch_norm:
                h = self.batch_norm2[ntype](h)

            new_h_dict[ntype] = h

        return new_h_dict


class GraphConv1(nn.Module):
    def __init__(self, device, n_layers, node_dims, hidden_dim, out_dim, n_heads, dropout):
        super(GraphConv1, self).__init__()

        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True

        self.node_types = ['metabolite', 'microbe']

        self.linear_h = nn.ModuleDict({
            ntype: nn.Linear(node_dims[ntype], hidden_dim)
            for ntype in self.node_types
        })

        self.layers = nn.ModuleList([
            HeteroGraphConvLayer(
                self.node_types,
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
            HeteroGraphConvLayer(
                self.node_types,
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

        h_dict = {}

        for ntype in self.node_types:
            h = g.ndata['h'][ntype].float().to(self.device)
            h = self.linear_h[ntype](h)
            h_dict[ntype] = h

        for conv in self.layers:
            h_dict = conv(g, h_dict)

        return h_dict

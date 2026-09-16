import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot

class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 residual: bool, dropout: float = 0.6, slope: float = 0.2, activation: nn.Module = nn.ELU()):
        super(GraphAttentionLayer, self).__init__(aggr='mean', node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = n_heads
        self.residual = residual

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=slope)
        self.activation = activation

        self.feat_lin = Linear(in_features, out_features * n_heads, bias=True, weight_initializer='glorot')
        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, out_features))

        # use 'residual' parameters to instantiate residual structure
        if residual:
            self.proj_r = Linear(in_features, out_features, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('proj_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attn_vec)

        self.feat_lin.reset_parameters()
        if self.proj_r is not None:
            self.proj_r.reset_parameters()

    def forward(self, x, edge_idx, size=None):
        # normalize input feature matrix
        x = self.feat_dropout(x)

        x_r = x_l = self.feat_lin(x).view(-1, self.heads, self.out_features)

        # calculate normal transformer components Q, K, V
        # The existing layer uses plain mean aggregation (no message override).
        # Sparse multiplication avoids materializing E x heads x features.
        if edge_idx.layout == torch.sparse_coo:
            aggregation = edge_idx
        else:
            src, dst = edge_idx
            degree = torch.bincount(dst, minlength=x.shape[0]).clamp(min=1)
            aggregation = torch.sparse_coo_tensor(
                torch.stack((dst, src)), 1.0 / degree[dst].to(x.dtype),
                (x.shape[0], x.shape[0]), device=x.device).coalesce()
        output = torch.sparse.mm(aggregation, x_l.reshape(x.shape[0], -1))
        output = output.view(x.shape[0], self.heads, self.out_features)

        if self.proj_r is not None:
            output = (output.transpose(0, 1) + self.proj_r(x)).transpose(1, 0)

        # output = self.activation(output)
        output = output.mean(dim=1)

        return output

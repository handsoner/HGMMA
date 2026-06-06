import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_meta, gt_net_micro, gt_meta_micro
from model.graph_transformer_layer import GraphTransformerLayer
from torch_geometric.nn.dense.linear import Linear
from model.GAT import GraphAttentionLayer
from data_preprocess import *
import torch.nn.functional as F
from torch_geometric.nn import conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args
        het_dim = args.meta_number + args.micro_number
        # self.liner = nn.Linear(867, args.gt_out_dim)

        # self.gt_meta = gt_net_meta.GraphConv1(device, args.gt_layer, args.meta_number, args.gt_out_dim,
        #                                             args.gt_out_dim, args.gt_head, args.dropout)
        # self.gt_micro = gt_net_micro.GraphConv1(device, args.gt_layer, args.micro_number, args.gt_out_dim,
        #                                               args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_meta = gt_net_meta.GraphTransformer(device, args.gt_layer, args.meta_number, args.gt_out_dim,
                                                    args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_micro = gt_net_micro.GraphTransformer(device, args.gt_layer, args.micro_number, args.gt_out_dim,
                                                      args.gt_out_dim, args.gt_head, args.dropout)


        class CustomSequential(torch.nn.Sequential):
            def reset_parameters(self):
                for layer in self:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

        self.proj = CustomSequential(
            torch.nn.Linear(het_dim, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128, bias=True)
        )

        if self.proj is not None:
            self.proj.reset_parameters()
        if self.proj is not None:
            self.proj.reset_parameters()
        self.gat = GraphAttentionLayer(het_dim, 128, 3, residual=True)  #图注意力网络
        self.g = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_dim, bias=False),
                               nn.ReLU(inplace=True)).to(device)

        #新加入的代码
        self.hgms_block = HGMSBlock(dim=self.args.out_ft, tau=0.7, topk=20, alpha=0.1, beta=0.1).to(device)


        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim*2, 1024),  # + 256
            # nn.Linear(args.gt_out_dim, 1024),  # 消融
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, meta_meta_graph, micro_micro_graph, het_mat, edge_idx, adj_mat,
                sample):
        # 1. Homogeneous similarity graph representations
        # 这是同构图部分
        meta_sim = self.gt_meta(meta_meta_graph)   #直接进入Transformer提前单图的特征，代谢物
        micro_sim = self.gt_micro(micro_micro_graph)    #直接进入Transformer提前单图的特征，微生物

        # meta_sim = self.gt_meta(meta_meta_graph.ndata['meta_sim'].float().to(device))   #消融Transformer
        # micro_sim = self.gt_micro(micro_micro_graph.ndata['micro_sim'].float().to(device))  #消融Transformer

        # 2. Original heterogeneous inputs
        #这是异构图部分
        cnn_embd_hetro = self.proj(het_mat) if self.proj is not None else het_mat
        gat_embd = self.gat(het_mat, edge_idx)  #gat是图Transformer网络

        z_proj = cnn_embd_hetro
        z_gat = gat_embd

        # 3. HGMS-style replacement for HERO
        emb_het, emb_hom, loss_hgms = self.hgms_block(z_proj, z_gat, adj_mat)


        # 4. HGMS loss replaces HERO consistency/specificity loss
        loss_h = loss_hgms

        # 5. Fuse HGMS two views
        h_concat = torch.cat((emb_het, emb_hom), 1) #异构图的同质和异质图特征进行拼接
        meta_x = h_concat[:self.args.meta_number]   #拿走前1596个代谢物特征
        micro_x = h_concat[self.args.meta_number:]  #拿走剩下的微生物特征

        # 6. Replace vanilla contrastive_loss with HGMS weighted contrastive loss
        S_all = self.hgms_block.build_connection_strength(adj_mat)
        S_meta = S_all[:self.args.meta_number, :self.args.meta_number]
        S_micro = S_all[self.args.meta_number:, self.args.meta_number:]


        cl_loss2 = self.hgms_block.weighted_contrastive_loss(meta_x, meta_sim, S_meta)
        cl_loss3 = self.hgms_block.weighted_contrastive_loss(micro_x, micro_sim, S_micro)
        loss = loss_h + 0.5 * cl_loss2 + 0.5 * cl_loss3

        # 7.final prediction
        meta = torch.cat((meta_sim, meta_x), dim=1)
        micro = torch.cat((micro_sim, micro_x), dim=1)
        # meta = meta_sim #消融异构图
        # micro = micro_sim   #消融异构图
        meta_micro_embedding = torch.mul(meta[sample[:, 0]], micro[sample[:, 1]])
        # meta_micro_embedding = torch.cat((meta[sample[:, 0]], micro[sample[:, 1]], torch.mul(meta[sample[:, 0]], micro[sample[:, 1]]), torch.abs(meta[sample[:, 0]] - micro[sample[:, 1]])), dim=1) #自己的改动，1. 拼接 + 差异 + 乘积联合建模

        output = self.mlp(meta_micro_embedding)
        # loss = torch.tensor(0.0, device=output.device)  #消融所有对比损失
        return loss, output

#新引入的异构图算法HGMS
class HGMSBlock(nn.Module):
    """
    A lightweight HGMS-inspired block for replacing HERO in GHTMDA.

    Core ideas:
    1. Connection strength: construct a soft positive matrix S from the heterogeneous adjacency.
    2. Multi-view self-expression: use one view to reconstruct another view through a learnable self-expression matrix.
    3. Weighted contrastive learning: use S as soft positive weights instead of only diagonal positives.

    Inputs:
        z_proj: [N, d], projection view from het_mat
        z_gat:  [N, d], GAT view from heterogeneous graph
        adj_mat:[N, N], heterogeneous bipartite adjacency matrix

    Outputs:
        z_cs:   [N, d], connection-strength enhanced view
        z_se:   [N, d], self-expression enhanced view
        loss:   scalar, HGMS-style auxiliary loss
    """

    def __init__(self, dim, tau=0.7, topk=20, alpha=0.1, beta=0.1):
        super(HGMSBlock, self).__init__()

        self.dim = dim
        self.tau = tau
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.cs_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        self.se_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def build_connection_strength(self, adj_mat):
        """
        Build connection-strength matrix S.

        For a bipartite graph adjacency:
            adj_mat = [[0, A],
                       [A.T, 0]]

        Direct edges only capture cross-type relations.
        Two-hop structure captures same-type homophily:
            microbe-metabolite-microbe
            metabolite-microbe-metabolite

        We combine:
            direct connection + two-hop connection + self-loop
        """

        if not isinstance(adj_mat, torch.Tensor):
            A = torch.tensor(adj_mat, dtype=torch.float32, device=device)
        else:
            A = adj_mat.float().to(device)

        N = A.shape[0]
        I = torch.eye(N, device=A.device)

        # Direct connection strength
        A_direct = A + I

        # Two-hop connection strength
        A_two_hop = torch.matmul(A, A)
        A_two_hop = A_two_hop / (A_two_hop.max().clamp(min=1.0))

        # Combine direct and two-hop structural strength
        S = A_direct + A_two_hop

        # Remove extreme scale
        S = S / S.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return S
        # return torch.eye(N, device=A.device)    #消融强度增强模块

    def topk_sparsify(self, score):
        """
        Keep top-k self-expression coefficients for each node.
        """
        if self.topk is None or self.topk <= 0 or self.topk >= score.shape[1]:
            return score

        topk_val, topk_idx = torch.topk(score, self.topk, dim=1)
        sparse_score = torch.full_like(score, float("-inf"))
        sparse_score.scatter_(1, topk_idx, topk_val)
        return sparse_score

    def multi_view_self_expression(self, z_query, z_key, S):
        """
        Multi-view self-expression.

        z_query: one view, e.g. z_proj
        z_key:   another view, e.g. z_gat

        C is a self-expression matrix guided by connection strength S.
        z_se = C @ value(z_key)
        """

        q = F.normalize(self.q_proj(z_query), dim=1)
        k = F.normalize(self.k_proj(z_key), dim=1)
        v = self.v_proj(z_key)

        score = torch.matmul(q, k.t()) / self.tau

        # Connection strength as structural prior.
        # If S[i, j] is large, node j is more likely to be a reliable positive neighbor of node i.
        score = score + torch.log(S + 1e-8)

        score = self.topk_sparsify(score)
        C = F.softmax(score, dim=1)

        z_se = torch.matmul(C, v)

        return z_se, C

    def weighted_contrastive_loss(self, z1, z2, S):
        """
        HGMS-style weighted contrastive loss.

        Different from the original diagonal-only contrastive_loss:
            positive pairs are not only (i, i),
            but also structurally reliable neighbors weighted by S[i, j].

        z1, z2: [N, d]
        S:      [N, N], soft positive weights
        """

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim = torch.matmul(z1, z2.t()) / self.tau
        exp_sim = torch.exp(sim)

        # Normalize S row-wise as soft positive distribution
        pos_weight = S / S.sum(dim=1, keepdim=True).clamp(min=1e-8)

        numerator = (exp_sim * pos_weight).sum(dim=1)
        denominator = exp_sim.sum(dim=1).clamp(min=1e-8)

        loss = -torch.log(numerator / denominator.clamp(min=1e-8)).mean()

        return loss

    def forward(self, z_proj, z_gat, adj_mat):
        S = self.build_connection_strength(adj_mat)

        # View 1: connection-strength enhanced heterogeneous representation
        z_cs = torch.matmul(S, z_gat)
        z_cs = self.cs_proj(z_cs)

        # View 2: multi-view self-expression representation
        z_se_raw, C = self.multi_view_self_expression(z_proj, z_gat, S)     #消融多视图注释这一行
        # z_se = self.se_proj(z_proj)   #消融多视图

        z_se = self.se_proj(z_se_raw)
        # HGMS-style losses
        loss_cl = self.weighted_contrastive_loss(z_cs, z_se, S)

        # Self-expression reconstruction: z_se should retain information from z_proj
        loss_recon = F.mse_loss(z_se, z_proj)

        # Connection-strength consistency: learned C should not deviate too far from S
        loss_conn = F.mse_loss(C, S)
        # loss_conn = torch.tensor(0.0, device=z_proj.device)

        loss_hgms = loss_cl + self.alpha * loss_recon + self.beta * loss_conn

        return z_cs, z_se, loss_hgms
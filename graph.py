import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_size, hidden_size, graph_drop):
        super(GraphConvolutionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)

        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

    def forward(self, input):
        nodes_embed, node_adj = input
        h = torch.matmul(nodes_embed, self.W.unsqueeze(0))
        sum_nei = torch.zeros_like(h)
        sum_nei += torch.matmul(node_adj, h)
        degs = torch.sum(node_adj, dim=-1).float().unsqueeze(dim=-1)
        norm = 1.0 / degs
        dst = sum_nei * norm + self.bias
        out = self.drop(torch.relu(dst))
        return nodes_embed + out, node_adj


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key):
    N_bt, h, N_nodes, _ = query.shape
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    return scores

class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, edges, in_features: int, out_features: int, n_heads: int, dropout=0.0):
        super().__init__()

        self.h = n_heads
        self.d_k = out_features // n_heads
        self.edges = edges
        self.linear_layers = nn.ModuleList()
        for i in range(len(edges)):
            self.linear_layers.append(clones(nn.Linear(in_features, out_features), 2))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor,supervise_edge_type = None):
            N_bt, N_nodes, _ = h.shape
            adj_mat = adj_mat.unsqueeze(1)
            scores = torch.zeros(N_bt, self.h, N_nodes, N_nodes, device=h.device)

            supervised_attn = None
            if supervise_edge_type is not None:
                if supervise_edge_type not in self.edges:
                    raise ValueError(f"Invalid edge type: {supervise_edge_type}")
                supervise_idx = self.edges.index(supervise_edge_type)
                supervise_adj_val = supervise_idx + 1 
            

            for edge in range(len(self.edges)):
                q, k = [l(x).view(N_bt, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                        zip(self.linear_layers[edge], (h, h))]
                raw_score = attention(q, k)  # (B, H, N, N)
                mask = (adj_mat != edge + 1)
                masked_score = raw_score.masked_fill(mask, -1e9)  
                scores = torch.where(mask, scores, scores + masked_score) 

                if supervise_edge_type is not None and edge == supervise_idx:
                    mask_supervised = (adj_mat != supervise_adj_val)
                    score_for_supervision = raw_score.masked_fill(mask_supervised, -1e9)
                    supervised_attn = F.softmax(score_for_supervision, dim=-1)                 


            scores = scores.masked_fill(scores == 0, -1e9)
            scores = self.dropout(scores)
            attn = F.softmax(scores, dim=-1)

            if supervise_edge_type is not None:
                return attn.transpose(0, 1), supervised_attn.transpose(0, 1)
            else:
                return attn.transpose(0, 1)

class  AttentionGCNLayer(nn.Module):
    def __init__(self, edges,input_size, nhead=2, graph_drop=0.0, iters=2, attn_drop=0.0,beta =0.3):
        super(AttentionGCNLayer, self).__init__()
        self.nhead = nhead
        self.beta = beta 
        self.graph_attention = MultiHeadDotProductAttention(edges,input_size, input_size, self.nhead, attn_drop)
        self.gcn_layers = nn.Sequential(
            *[GraphConvolutionLayer(input_size, input_size, graph_drop) for _ in range(iters)])
        self.blocks = nn.ModuleList([self.gcn_layers for _ in range(self.nhead)])

        self.aggregate_W = nn.Linear(input_size * nhead, input_size)

    def forward(self, nodes_embed, node_adj,supervise_edge_type=None):
        output = []
        if supervise_edge_type is not None:
            graph_attention, supervised_attn = self.graph_attention(
                nodes_embed, node_adj, supervise_edge_type=supervise_edge_type
            )
        else:
            graph_attention = self.graph_attention(nodes_embed, node_adj)
            supervised_attn = None

        for cnt in range(0, self.nhead):
            hi, _ = self.blocks[cnt]((nodes_embed, graph_attention[cnt]))
            output.append(hi)

        output = torch.cat(output, dim=-1)
        output = self.aggregate_W(output)           # (B, N, D) 
        return output, graph_attention , supervised_attn

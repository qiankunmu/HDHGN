import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import HeteroLinear, MLP
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from models.layers import HeteroEmbedding, HDHGConv


class HDHGN(nn.Module):
    def __init__(self, num_types, vocab_sizes, edge_vocab_size, embed_size, dim_size, num_layers, num_edge_heads,
                 num_node_heads, num_heads, feed_sizes, dropout_rate):
        super(HDHGN, self).__init__()
        self.num_types = num_types
        self.vocab_sizes = vocab_sizes
        self.edge_vocab_size = edge_vocab_size
        self.embed_size = embed_size
        self.dim_size = dim_size
        self.num_layers = num_layers
        self.num_edge_heads = num_edge_heads
        self.num_node_heads = num_node_heads
        self.num_heads = num_heads
        self.feed_sizes = feed_sizes
        self.dropout_rate = dropout_rate

        self.embedding = HeteroEmbedding(self.num_types, self.vocab_sizes, self.embed_size)
        self.hetero_linear = HeteroLinear(self.embed_size, self.dim_size, self.num_types)
        self.edge_embedding = nn.Embedding(self.edge_vocab_size, self.embed_size, padding_idx=0)

        self.HPHG = nn.ModuleList(
            [HDHGConv(self.dim_size, self.num_edge_heads, self.num_node_heads, self.dropout_rate) for i in
             range(self.num_layers)])
        self.attn = nn.Parameter(
            torch.Tensor(1, self.num_heads, self.dim_size // self.num_heads))
        self.mlp = MLP(self.feed_sizes, act="elu", dropout=self.dropout_rate, batch_norm=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, types, edge_types, edge_in_indexs, edge_out_indexs, edge_in_out_indexs, edge_in_out_head_tail, batch):
        # x, types [num_nodes] edge_types [num_edges] edge_in_indexs [2, num_nodes] edge_out_indexs [2, num_edges] edge_in_out_indexs [2, num_nodeedges]
        x = self.embedding(x, types)
        # x [num_nodes, embed_size]
        x = self.hetero_linear(x, types)
        # x [num_nodes, dim_size]
        edge_attr = self.edge_embedding(edge_types)
        # edge_attr [num_edges, dim_size]
        for i in range(self.num_layers):
            x = self.HPHG[i](x, edge_attr, edge_in_indexs, edge_out_indexs, edge_in_out_indexs, edge_in_out_head_tail, batch)

        x = x.reshape(-1, self.num_heads, self.dim_size // self.num_heads)
        attn = (self.attn * x).sum(dim=-1)
        # attn [num_nodes, num_heads]
        attn_score = softmax(attn, batch)
        attn_score = attn_score.unsqueeze(-1)
        # attn_score = self.dropout(attn_score)
        x = x * attn_score
        v = scatter_add(x, batch, 0)
        # out [batch_size, num_heads, head_size]
        v = v.reshape(-1, self.dim_size)

        out = self.mlp(v)
        return out

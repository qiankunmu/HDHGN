import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import math


class HeteroEmbedding(nn.Module):
    def __init__(self, num_types, vocab_sizes, embed_size):
        super(HeteroEmbedding, self).__init__()
        self.num_types = num_types
        self.vocab_sizes = vocab_sizes
        self.embed_size = embed_size

        self.embedding = nn.ModuleList(
            [nn.Embedding(self.vocab_sizes[i], self.embed_size, padding_idx=0) for i in range(self.num_types)])

    def forward(self, x, types):
        # x, types [num_nodes]
        out = x.new_empty(x.size(0), self.embed_size, dtype=torch.float)
        for i, embedding in enumerate(self.embedding):
            mask = types == i
            out[mask] = embedding(x[mask])
        # out [num_nodes, embed_size]
        return out


class HDHGConv(MessagePassing):
    def __init__(self, dim_size, num_edge_heads, num_node_heads):
        super(HDHGConv, self).__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.dim_size = dim_size
        self.num_edge_heads = num_edge_heads
        self.num_node_heads = num_node_heads

        self.W_key1 = nn.Parameter(torch.Tensor(1, self.num_edge_heads, self.dim_size // self.num_edge_heads,
                                                self.dim_size // self.num_edge_heads))
        self.W_query1 = nn.Parameter(torch.Tensor(1, self.num_edge_heads, self.dim_size // self.num_edge_heads,
                                                  self.dim_size // self.num_edge_heads))
        self.W_vector1 = nn.Parameter(torch.Tensor(1, self.num_edge_heads, self.dim_size // self.num_edge_heads,
                                                   self.dim_size // self.num_edge_heads))

        self.edge_linear = nn.Linear(self.dim_size, self.dim_size)

        self.head_tail_linear = HeteroLinear(self.dim_size, self.dim_size, 2)

        self.to_head_tail_linear = HeteroLinear(self.dim_size, self.dim_size, 2)

        self.W_key2 = nn.Parameter(torch.Tensor(1, self.num_node_heads, self.dim_size // self.num_node_heads,
                                                self.dim_size // self.num_node_heads))
        self.W_query2 = nn.Parameter(torch.Tensor(1, self.num_node_heads, self.dim_size // self.num_node_heads,
                                                  self.dim_size // self.num_node_heads))
        self.W_vector2 = nn.Parameter(torch.Tensor(1, self.num_node_heads, self.dim_size // self.num_node_heads,
                                                   self.dim_size // self.num_node_heads))

        self.u1 = nn.Linear(self.dim_size, self.dim_size)
        self.u2 = nn.Linear(self.dim_size, self.dim_size)

        self.norm = GraphNorm(self.dim_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_key1)
        nn.init.xavier_uniform_(self.W_query1)
        nn.init.xavier_uniform_(self.W_vector1)

        nn.init.xavier_uniform_(self.W_key2)
        nn.init.xavier_uniform_(self.W_query2)
        nn.init.xavier_uniform_(self.W_vector2)

    def forward(self, x, edge_attr, edge_in_out_indexs, edge_in_out_head_tail, batch):
        # x [num_nodes, dim_size] edge_attr [num_edges, dim_size] edge_in_out_indexs [2, num_nodeedges] edge_in_out_head_tail [num_nodeedges]
        hyperedges = self.edge_updater(edge_in_out_indexs.flip([0]), x=x, edge_attr=edge_attr,
                                       edge_in_out_head_tail=edge_in_out_head_tail)
        # hyperedges [num_edges, dim_size]
        edge_attr_out = self.edge_linear(edge_attr)
        hyperedges = hyperedges + edge_attr_out
        out = self.propagate(edge_in_out_indexs, x=x, hyperedges=hyperedges,
                             edge_in_out_head_tail=edge_in_out_head_tail, batch=batch)
        return out

    def edge_update(self, edge_index=None, x_j=None, edge_attr_i=None, edge_in_out_head_tail=None):
        m = self.head_tail_linear(x_j, edge_in_out_head_tail)
        m = m.reshape(-1, self.num_edge_heads, 1, self.dim_size // self.num_edge_heads)
        edge_attr_i = edge_attr_i.reshape(-1, self.num_edge_heads, 1, self.dim_size // self.num_edge_heads)
        # m, edge_attr_j [num_nodeedges, num_edge_heads, 1, head_size]
        query = (self.W_query1 * edge_attr_i).sum(dim=-1)
        key = (self.W_key1 * m).sum(dim=-1)
        vector = (self.W_vector1 * m).sum(dim=-1)
        # query, key, vector [num_nodeedges, num_edge_heads, head_size]
        attn = (query * key).sum(dim=-1)
        attn = attn / math.sqrt(self.dim_size // self.num_edge_heads)
        # attn [num_nodeedges, num_edge_heads]
        attn_score = softmax(attn, edge_index[1])
        attn_score = attn_score.unsqueeze(-1)
        # attn_score [num_nodeedges, num_edge_heads, 1]
        out = vector * attn_score
        # out [num_nodeedges, num_edge_heads, head_size]
        out = scatter_add(out, edge_index[1], 0)
        # out [num_edges, num_edge_heads, head_size]
        out = out.reshape(-1, self.dim_size)

        return out

    def message(self, edge_index=None, x_i=None, hyperedges_j=None, edge_in_out_head_tail=None):
        m = self.to_head_tail_linear(hyperedges_j, edge_in_out_head_tail)
        m, x_i = m.reshape(-1, self.num_node_heads, 1, self.dim_size // self.num_node_heads), \
                 x_i.reshape(-1, self.num_node_heads, 1, self.dim_size // self.num_node_heads)
        # m, x_i [num_nodeedges, num_node_heads, 1, head_size]
        query = (self.W_query2 * x_i).sum(dim=-1)
        key = (self.W_key2 * m).sum(dim=-1)
        vector = (self.W_vector2 * m).sum(dim=-1)
        # query, key, vector [num_nodeedges, num_node_heads, head_size]
        attn = (query * key).sum(dim=-1)
        # attn [num_nodeedges, num_nodes_heads]
        attn = attn / math.sqrt(self.dim_size // self.num_node_heads)
        attn_score = softmax(attn, edge_index[1])
        attn_score = attn_score.unsqueeze(-1)
        # attn_score [num_nodeedges, num_node_heads, 1]
        out = vector * attn_score
        # out [num_nodeedges, num_node_heads, head_size]

        return out

    def update(self, inputs, x=None, batch=None):
        inputs = inputs.reshape(-1, self.dim_size)
        # x, inputs [num_nodes, dim_size]
        inputs = self.u2(inputs)
        x = self.u1(x)
        out = inputs + x
        out = self.norm(out, batch)
        out = F.elu(out)
        return out

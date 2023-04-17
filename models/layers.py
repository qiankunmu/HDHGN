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


class LinearMatrix(nn.Module):
    def __init__(self, in_dim_size, out_num_dim, out_dim_size):
        super(LinearMatrix, self).__init__()
        self.in_dim_size = in_dim_size
        self.out_num_dim = out_num_dim
        self.out_dim_size = out_dim_size

        self.linear = nn.Linear(self.in_dim_size, self.out_num_dim * self.out_dim_size, bias=False)

    def forward(self, input):
        output = self.linear(input)
        output = output.reshape(-1, self.out_num_dim, self.out_dim_size)
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()


class HDHGConv(MessagePassing):
    def __init__(self, dim_size, num_edge_heads, num_node_heads, dropout_rate, negative_slope=0.2):
        super(HDHGConv, self).__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.dim_size = dim_size
        self.num_edge_heads = num_edge_heads
        self.num_node_heads = num_node_heads
        self.dropout_rate = dropout_rate
        self.negative_slope = negative_slope

        self.edge_in_linear = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.edge_out_linear = LinearMatrix(self.dim_size, self.dim_size, self.dim_size)
        self.W = nn.Parameter(torch.Tensor(1, self.num_edge_heads, self.dim_size // self.num_edge_heads,
                                               self.dim_size // self.num_node_heads))
        self.attn = nn.Parameter(torch.Tensor(1, self.num_edge_heads, 2 * (self.dim_size // self.num_edge_heads)))

        self.head_linear = nn.Linear(self.dim_size, self.dim_size)
        self.tail_linear = nn.Linear(self.dim_size, self.dim_size)

        self.to_head_tail_linear = HeteroLinear(self.dim_size, self.dim_size, 2)

        self.W_key = nn.Parameter(torch.Tensor(1, self.num_node_heads, self.dim_size // self.num_node_heads,
                                               self.dim_size // self.num_node_heads))
        self.W_query = nn.Parameter(torch.Tensor(1, self.num_node_heads, self.dim_size // self.num_node_heads,
                                                 self.dim_size // self.num_node_heads))

        self.dropout = nn.Dropout(self.dropout_rate)

        self.parent_linear = nn.Linear(self.dim_size, self.dim_size)
        self.children_linear = nn.Linear(self.dim_size, self.dim_size)

        self.norm = GraphNorm(self.dim_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.W_key)
        nn.init.xavier_uniform_(self.W_query)

    def forward(self, x, edge_attr, edge_in_indexs, edge_out_indexs, edge_in_out_indexs, edge_in_out_head_tail, batch):
        # x [num_nodes, dim_size] edge_attr [num_edges, dim_size] edge_in_indexs [2, num_nodes] edge_out_indexs [2, num_edges] edge_in_out_indexs [2, num_nodeedges] edge_in_out_tail [num_nodeedges]
        edge_in_attr = self.edge_in_linear(edge_attr)
        hyperedge_tails = self.edge_updater(edge_in_indexs, x=x, edge_in_attr=edge_in_attr)
        # hyperedge_tails [num_edges, dim_size]
        hyperedge_heads = x[edge_out_indexs[1]]
        # hyperedge_heads [num_edges, dim_size]
        hyperedges = self.tail_linear(hyperedge_tails) + self.head_linear(hyperedge_heads)
        edge_out_attr = self.edge_out_linear(edge_attr)
        # edge_out_attr [num_edges, dim_size, dim_size]
        out = self.propagate(edge_in_out_indexs, x=x, hyperedges=hyperedges, edge_out_attr=edge_out_attr, edge_in_out_head_tail=edge_in_out_head_tail, batch=batch)
        return out


    def edge_update(self, edge_index=None, x_j=None, edge_in_attr_i=None):
        x_j = x_j.reshape(-1, self.num_edge_heads, 1, self.dim_size // self.num_edge_heads)
        edge_in_attr_i = edge_in_attr_i.reshape(-1, self.num_edge_heads, 1, self.dim_size // self.num_edge_heads)
        # x, edge_in_attr [num_nodes, num_edge_heads, 1, head_size]
        x_j = (self.W * x_j).sum(dim=-1)
        edge_in_attr_i = (self.W * edge_in_attr_i).sum(dim=-1)
        # x, edge_in_attr [num_nodes, num_edge_heads, head_size]
        attn = (torch.cat([x_j, edge_in_attr_i], dim=-1) * self.attn).sum(dim=-1)
        # attn [num_nodes, num_edge_heads]
        attn = F.leaky_relu(attn, self.negative_slope)
        attn_score = softmax(attn, edge_index[1])
        attn_score = attn_score.unsqueeze(-1)
        # attn_score = self.dropout(attn_score)
        # attn_score [num_nodes, num_edge_heads, 1]
        out = x_j * attn_score
        # out [num_nodes, num_edge_heads, head_size]
        out = scatter_add(out, edge_index[1], 0)
        # out [num_edges, num_edge_heads, head_size]
        out = out.reshape(-1, self.dim_size)
        return out


    def message(self, edge_index=None, x_i=None, hyperedges_j=None, edge_out_attr_j=None, edge_in_out_head_tail=None):
        m = (hyperedges_j.unsqueeze(1) * edge_out_attr_j).sum(dim=-1)
        # m [num_edges, dim_size]
        m = self.to_head_tail_linear(m, edge_in_out_head_tail)
        m, x_i = m.reshape(-1, self.num_node_heads, 1, self.dim_size // self.num_node_heads), \
                 x_i.reshape(-1, self.num_node_heads, 1, self.dim_size // self.num_node_heads)
        # [num_edges, num_node_heads, 1, head_size]
        query = (self.W_query * x_i).sum(dim=-1)
        key = (self.W_key * m).sum(dim=-1)
        # query, key [num_edges, num_node_heads, head_size]
        attn = (query * key).sum(dim=-1)
        # attn [num_edges, num_nodes_heads]
        attn = attn / math.sqrt(self.dim_size // self.num_node_heads)
        attn_score = softmax(attn, edge_index[1])
        attn_score = attn_score.unsqueeze(-1)
        # attn_score [num_edges, num_node_heads, 1]
        # attn_score = self.dropout(attn_score)
        m = m.squeeze(2)
        out = m * attn_score
        return out

    def update(self, inputs, x=None, batch=None):
        inputs = inputs.reshape(-1, self.dim_size)
        # inputs [num_nodes, dim_size]
        inputs = self.children_linear(inputs)
        x = self.parent_linear(x)
        out = inputs + x
        out = self.norm(out, batch)
        out = F.elu(out)
        # out = self.dropout(out)
        return out
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
def Truncated_initializer(m):
    # sample u1:
    size = m.size()
    u1 = torch.rand(size)*(1-np.exp(-2)) + np.exp(-2)
    # sample u2:
    u2 = torch.rand(size)
    # sample the truncated gaussian ~TN(0,1,[-2,2]):
    z = torch.sqrt(-2*torch.log(u1)) * torch.cos(2*np.pi*u2)
    m.data = z


class GraphConvolution_adj_learnable(nn.Module):
    def __init__(self, metagraph_vertex_num, hidden_dim, sparse_inputs=False, act=nn.Tanh(), bias=True, dropout=0.6):
        super(GraphConvolution_adj_learnable, self).__init__()
        self.active_function = act
        self.dropout_rate = dropout
        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.metagraph_vertex_num = metagraph_vertex_num
        self.A = nn.Parameter(torch.ones(size=(self.metagraph_vertex_num, self.metagraph_vertex_num)))
        self.W = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))
        Truncated_initializer(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.b = None
        self.device = torch.device('cuda')

    def normalize(self, m):
        rowsum = torch.sum(m, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.diag(r_inv).float()

        m_norm = torch.mm(r_mat_inv, m)
        m_norm = torch.mm(m_norm, r_mat_inv)
        return m_norm

    def forward(self, inputs):
        x = inputs
        x = self.dropout(x)
        adj_norm = self.normalize(self.A)

        pre_sup = torch.matmul(x, self.W)
        output = torch.matmul(adj_norm, pre_sup)


        if self.bias:
            output += self.b
        if self.active_function is not None:
            return self.active_function(output)
        else:
            return output

class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, sparse_inputs=False, act=nn.Tanh(), bias=True, dropout=0.6):
        super(GraphConvolution, self).__init__()
        self.active_function = act
        self.dropout_rate = dropout
        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))
        Truncated_initializer(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.b = None
        self.device = torch.device('cuda')

    def forward(self, inputs, adj):
        x = inputs
        x = self.dropout(x)
        node_size = adj.size(0)
        I = torch.eye(node_size, requires_grad=False).to(self.device)
        adj = adj + I
        D = torch.diag(torch.sum(adj, dim=1, keepdim=False))
        adj = torch.matmul(torch.inverse(D), adj)
        pre_sup = torch.matmul(x, self.W)
        output = torch.matmul(adj, pre_sup)

        if self.bias:
            output += self.b
        if self.active_function is not None:
            return self.active_function(output)
        else:
            return output



class OrthogonalMetaGraph(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(OrthogonalMetaGraph, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)

        gate_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num
        self.proto_graph_vertex_num = proto_graph_vertex_num
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim))
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid())
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid())
        self.device = torch.device('cuda')
        self.meta_GCN = GraphConvolution(self.hidden_dim).to(self.device)
        self.MSE = nn.MSELoss(reduce='mean')



    def forward(self, inputs):
        batch_size = inputs.size(0)
        protos = inputs
        meta_graph = self._construct_graph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)
        proto_graph = self._construct_graph(protos, protos).to(self.device)
        m, n = protos.size(0), self.meta_graph_vertex.size(0)
        xx = torch.pow(protos, 2).sum(1, keepdim=True).expand(m,n)
        yy = torch.pow(self.meta_graph_vertex, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(mat1=protos, mat2=self.meta_graph_vertex.t(), beta=1, alpha=-2)
        dist_square = dist.clamp(min=1e-6)
        cross_graph = self.softmax(
            (- dist_square / (2.0 * self.sigma))).to(self.device)
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph),  dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)
        representation = self.meta_GCN(feature, super_garph)

        # most_similar_index = torch.argmax(cross_graph, dim=1)
        normalized_meta = F.normalize(self.meta_graph_vertex)
        ccT = torch.mm(normalized_meta, normalized_meta.t())
        I = torch.eye(self.meta_graph_vertex_num, requires_grad=False).to(self.device)
        orthogonality = self.MSE(ccT, I)

        return representation[0:batch_size].to(self.device), orthogonality

    def _construct_graph(self, A, B):
        m = A.size(0)
        n = B.size(0)
        I = torch.eye(n, requires_grad=False).to(self.device)
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n)
        graph = graph.to(self.device) * (1 - I) + I
        return graph

class MetaGraph_fd(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(MetaGraph_fd, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)

        gate_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num
        self.proto_graph_vertex_num = proto_graph_vertex_num
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim))
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid())
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid())
        self.device = torch.device('cuda')
        self.meta_GCN = GraphConvolution(self.hidden_dim).to(self.device)
        self.MSE = nn.MSELoss(reduce='mean')
        self.register_buffer('meta_graph_vertex_buffer', torch.rand(self.meta_graph_vertex.size(), requires_grad=False))


    def forward(self, inputs):
        correlation_meta = self._correlation(self.meta_graph_vertex_buffer, self.meta_graph_vertex.detach())

        self.meta_graph_vertex_buffer = self.meta_graph_vertex.detach()

        batch_size = inputs.size(0)
        protos = inputs
        meta_graph = self._construct_graph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)
        proto_graph = self._construct_graph(protos, protos).to(self.device)
        m, n = protos.size(0), self.meta_graph_vertex.size(0)
        xx = torch.pow(protos, 2).sum(1, keepdim=True).expand(m,n)
        yy = torch.pow(self.meta_graph_vertex, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(mat1=protos, mat2=self.meta_graph_vertex.t(), beta=1, alpha=-2)
        dist_square = dist.clamp(min=1e-6)
        cross_graph = self.softmax(
            (- dist_square / (2.0 * self.sigma))).to(self.device)
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph),  dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)
        representation = self.meta_GCN(feature, super_garph)

        # most_similar_index = torch.argmax(cross_graph, dim=1)
        correlation_transfer_meta = self._correlation(representation[batch_size:].detach(), self.meta_graph_vertex.detach())


        correlation_protos = self._correlation(representation[0:batch_size].detach(), protos.detach())



        return representation[0:batch_size].to(self.device), [correlation_meta,correlation_transfer_meta, correlation_protos]

    def _construct_graph(self, A, B):
        m = A.size(0)
        n = B.size(0)
        I = torch.eye(n, requires_grad=False).to(self.device)
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n)
        graph = graph.to(self.device) * (1 - I) + I
        return graph

    def _correlation(self, A, B):
        similarity = F.cosine_similarity(A,B)
        similarity = torch.mean(similarity)
        return similarity



class FixedMetaGraph(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(FixedMetaGraph, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)

        gate_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num
        self.proto_graph_vertex_num = proto_graph_vertex_num
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim), requires_grad=False)
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid())
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid())
        self.device = torch.device('cuda')
        self.meta_GCN = GraphConvolution(self.hidden_dim).to(self.device)
        self.MSE = nn.MSELoss(reduce='mean')



    def forward(self, inputs):
        batch_size = inputs.size(0)
        protos = inputs
        meta_graph = self._construct_graph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)
        proto_graph = self._construct_graph(protos, protos).to(self.device)
        m, n = protos.size(0), self.meta_graph_vertex.size(0)
        xx = torch.pow(protos, 2).sum(1, keepdim=True).expand(m,n)
        yy = torch.pow(self.meta_graph_vertex, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(mat1=protos, mat2=self.meta_graph_vertex.t(), beta=1, alpha=-2)
        dist_square = dist.clamp(min=1e-6)
        cross_graph = self.softmax(
            (- dist_square / (2.0 * self.sigma))).to(self.device)
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph),  dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)
        representation = self.meta_GCN(feature, super_garph)

        # most_similar_index = torch.argmax(cross_graph, dim=1)
        normalized_transfered_meta = F.normalize(representation[batch_size:])
        normalized_meta = F.normalize(self.meta_graph_vertex)
        ccT = torch.mm(normalized_transfered_meta, normalized_transfered_meta.t())
        mmT = torch.mm(normalized_meta, normalized_meta.t())
        # I = torch.eye(self.meta_graph_vertex_num, requires_grad=False).to(self.device)
        correlation = self.MSE(ccT, mmT)

        return representation[0:batch_size].to(self.device), correlation

    def _construct_graph(self, A, B):
        m = A.size(0)
        n = B.size(0)
        I = torch.eye(n, requires_grad=False).to(self.device)
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n)
        graph = graph.to(self.device) * (1 - I) + I
        return graph


if __name__ == '__main__':
    model = OrthogonalMetaGraph(hidden_dim=2048, input_dim=2048, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128)

    x = torch.rand(64, 2048)
    representation, graph = model(x)
    print(representation.size())
    print(graph)


import torch
import argparse
import csv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, RGATConv, GAT, GATv2Conv, TransformerConv
from torch_geometric.data import Data, HeteroData
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph, add_self_loops, to_dense_batch
class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_relations, edge_dim, dropout, is_full=False):
        super(RGAT, self).__init__()
        # Initialize the layers
        # print(in_channels,hidden_channels,heads)
        self.cn1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout, heads=heads, bias=False)
        self.is_full = is_full
        #self.ln1 = torch.nn.Linear(hidden_channels, out_channels)
        self.edge_emb_layer = torch.nn.Embedding(num_relations, edge_dim)
        if hidden_channels*heads != out_channels:
            self.project_layer = torch.nn.Linear(hidden_channels*heads, out_channels, bias=False)
    def forward(self, x, edge_index, edge_type, return_attention_weights=False):
        edge_attr = self.edge_emb_layer(edge_type)
        if return_attention_weights:
            x, (alpha, beta) = self.cn1(x, edge_index, edge_attr=edge_attr, return_attention_weights=return_attention_weights)
        else:
            x = self.cn1(x, edge_index, edge_attr=edge_attr)
            if hasattr(self, 'project_layer'):
                x = self.project_layer(x)
        if return_attention_weights:
            return x, alpha, beta
        else:
            return x
import copy
class KGCodeReassign(torch.nn.Module):
    def __init__(self, args, edges_dict, c2ind, cm2ind):
        super(KGCodeReassign, self).__init__()
        # self.GATmodel = GAT(args["num_features"], args["hidden_channels"], args["embedding_dim"],
        #                     args["heads"]).to(args.device)
        self.RGATmodel = RGAT(args['attention_dim'], args['attention_dim']//args['use_multihead'], args['attention_dim'], args['use_multihead'],
                              num_relations=11, edge_dim=args['edge_dim'], dropout=args['rep_dropout']/2, is_full=len(c2ind)>50)
        self.original_code_num = len(c2ind)
        self.c2ind = copy.deepcopy(c2ind)
        self.cm2ind = copy.deepcopy(cm2ind)
        self.mcodes = []
        edges = [[],[]]
        edges_type = []
        for edge_pair in edges_dict.keys():
            if edges_dict[edge_pair] != 0:
                edges[0].append(self.cm2ind[edge_pair[0]] + self.original_code_num)
                edges[1].append(self.c2ind[edge_pair[1]])
            elif edges_dict[edge_pair] == 0:
                edges[0].append(self.c2ind[edge_pair[0]])
                edges[1].append(self.c2ind[edge_pair[1]])
            edges_type.append(edges_dict[edge_pair])
        self.mcodes = torch.nn.Parameter(torch.arange(0, len(cm2ind))+len(c2ind), requires_grad=False)
        self.edges = torch.nn.Parameter(torch.LongTensor(edges), requires_grad=False)
        self.edges_type = torch.nn.Parameter(torch.LongTensor(edges_type), requires_grad=False)
        self.ind2c = {v: k for k, v in self.c2ind.items()}
        self.ind2mc = {v+self.original_code_num: k for k, v in self.cm2ind.items()}
        self.ind2mc.update(self.ind2c)
        #self.ind2c.update(self.ind2mc)

    def forward(self, code_embeddings, mcode_embeddings, indices, return_attention_weights=False):
        #[B, C, E]
        batch_data = []
        edges_reals = []
        if len(code_embeddings.shape) == 2:
            if indices is not None:
                edges, edges_type = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=True)
                if return_attention_weights:
                    edges_real, _ = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=False)
                    edges_reals.append(edges_real)
            else:
                edges = self.edges
                edges_type = self.edges_type
            topk_code_embedding = torch.cat([code_embeddings, mcode_embeddings], dim=0)
            batch_data.append(Data(x=topk_code_embedding, edge_index=edges, edge_type=edges_type))
        else:
            for ind in range(len(code_embeddings)):
                if indices is not None:
                    edges, edges_type = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=True)
                    if return_attention_weights:
                        edges_real, _ = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=False)
                        edges_reals.append(edges_real)
                else:
                    edges = self.edges
                    edges_type = self.edges_type
                topk_code_embedding = torch.cat([code_embeddings[ind], mcode_embeddings[ind]], dim=0)
                batch_data.append(Data(x=topk_code_embedding, edge_index=edges, edge_type=edges_type))
        batch = Batch.from_data_list(batch_data)
        #batch_index = torch.cat(b_index, dim=0)
        if return_attention_weights:
            graph_embeddings_, edge_index_used, attention_weights = self.RGATmodel(batch.x, batch.edge_index, edge_type=batch.edge_type, return_attention_weights=return_attention_weights)
            edge_index_used = edge_index_used.view(2, -1, edges.shape[1])
            attention_weights = attention_weights.mean(dim=1)
            attention_weights = attention_weights.view(-1, edges.shape[1])
            delta = edge_index_used.shape[-1]
            attentions_rs = []
            for bid in range(code_embeddings.shape[0]):
                if indices is None:
                    current_edge_index = edge_index_used[:, bid, :] - delta*bid
                else:
                    current_edge_index = edges_reals[bid]
                current_attention_weights = attention_weights[bid, :]
                attentions_dict = {}
                for code_key in self.c2ind.keys():
                    code_id = self.c2ind[code_key]
                    index = (current_edge_index[1] == code_id)
                    if index.sum() > 0:
                        attentions_of_code = current_attention_weights[index]
                        #print(current_edge_index)
                        #print(attentions_of_code.shape, attentions_of_code.sum())
                        source_code_id = current_edge_index[0][index]
                        source_code_names = [self.ind2mc[int(i)] for i in source_code_id]
                        attentions_of_code = attentions_of_code.cpu().detach().numpy()
                        results = list(zip(source_code_names, attentions_of_code))
                        results = sorted(results, key=lambda x: x[1], reverse=True)[0:10]
                        if attentions_of_code.max() > 0.0015:
                            print(code_key, results)
                        attentions_dict[code_key] = results
                attentions_rs.append(attentions_dict)
        else:
            graph_embeddings_ = self.RGATmodel(batch.x, batch.edge_index, edge_type=batch.edge_type)
        if len(code_embeddings.shape) == 3:
            graph_embeddings = to_dense_batch(graph_embeddings_, batch.batch)[0]
            graph_embeddings = graph_embeddings[:, 0:code_embeddings.shape[1]]
        else:
            graph_embeddings = graph_embeddings_[0:code_embeddings.shape[0]]
        if return_attention_weights:
            return graph_embeddings, attentions_rs
        else:
            return graph_embeddings
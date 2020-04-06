import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv


class BaseRGCN(nn.Module):
    def __init__(self, node_attri, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model(node_attri)

    def build_model(self, node_attri):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer(node_attri)
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self, node_attri):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h
    
    
class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())
    
    
class EmbeddingLayerAttri(nn.Module):
    def __init__(self, node_attri):
        super(EmbeddingLayerAttri, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(node_attri))

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())
    
    
class RGCN(BaseRGCN):
    def build_input_layer(self, node_attri):
        if node_attri is not None:
            return EmbeddingLayerAttri(node_attri)
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        if idx==0:
            return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis", self.num_bases, activation=act, self_loop=True, dropout=self.dropout)
        return RelGraphConv(self.out_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)
    

class TrainModel(nn.Module):
    def __init__(self, node_attri, num_nodes, o_dim, num_rels, nlabel, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(TrainModel, self).__init__()
        
        if node_attri is None:
            self.rgcn = RGCN(node_attri, num_nodes, o_dim, o_dim, num_rels * 2, num_bases, num_hidden_layers, dropout, use_cuda)
        else:            
            self.rgcn = RGCN(node_attri, num_nodes, node_attri.shape[1], o_dim, num_rels * 2, num_bases, num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        
        if nlabel==0:
            self.supervised = False
            self.w_relation = nn.Parameter(torch.Tensor(num_rels, o_dim))
            nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
        else:
            self.supervised = True
            self.LinearLayer = torch.nn.Linear(o_dim, nlabel)

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        output = self.rgcn.forward(g, h, r, norm)
        if self.supervised:
            pred = self.LinearLayer(output)
        else:
            pred = None
        return output, pred

    def unsupervised_regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_unsupervised_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.unsupervised_regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss
    
    def supervised_regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))
    
    def get_supervised_loss(self, embed, matched_labels, matched_index, multi):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        if multi: 
            predict_loss = F.binary_cross_entropy(torch.sigmoid(embed[matched_index]), matched_labels)
        else:
            predict_loss = F.nll_loss(F.log_softmax(embed[matched_index]), matched_labels)
        reg_loss = self.supervised_regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss
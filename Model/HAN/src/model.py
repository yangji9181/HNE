import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class HomoAttLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout, alpha, device):
        super(HomoAttLayer, self).__init__()
        
        self.dropout = dropout
        self.device = device
        
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def forward(self, features, adj, target_len, neighbor_len, target_index_out):
        h = torch.mm(features, self.W)
        
        compare = torch.cat([h[adj[0]], h[adj[1]]], dim=1)
        e = self.leakyrelu(torch.matmul(compare, self.a).squeeze(1))
        
        attention = torch.full((target_len, neighbor_len), -9e15).to(self.device)
        attention[target_index_out, adj[1]] = e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)
    
    
class HomoAttModel(nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout, alpha, device, nheads, nlayer, neigh_por):
        super(HomoAttModel, self).__init__()
        
        self.neigh_por = neigh_por
        self.nlayer = nlayer
        self.dropout = dropout
        
        self.homo_atts = []
        for i in range(nlayer):
            
            if i==0: curr_in_dim = in_dim
            else: curr_in_dim = out_dim*nheads[i-1]
            layer_homo_atts = []
            
            for j in range(nheads[i]):
                layer_homo_atts.append(HomoAttLayer(curr_in_dim, out_dim, dropout, alpha, device).to(device))
                self.add_module('homo_atts_layer{}_head{}'.format(i,j), layer_homo_atts[j])
            self.homo_atts.append(layer_homo_atts)
                
    def sample(self, adj, samples):

        sample_list, adj_list = [samples], []
        for _ in range(self.nlayer):
            
            new_samples, new_adjs = set(sample_list[-1]), []
            for sample in sample_list[-1]:
                neighbor_size = adj[1][sample]
                nneighbor = int(self.neigh_por*neighbor_size)+1
                start = adj[1][:sample].sum()
                
                if neighbor_size<=nneighbor:
                    curr_new_samples = adj[0][start:start+neighbor_size]   
                else:
                    curr_new_samples = random.sample(adj[0][start:start+neighbor_size].tolist(), nneighbor)
                new_samples = new_samples.union(set(curr_new_samples))
                curr_new_adjs = np.stack(([sample]*len(curr_new_samples), curr_new_samples), axis=-1).tolist()
                curr_new_adjs.append([sample, sample])
                new_adjs.append(curr_new_adjs)

            sample_list.append(np.array(list(new_samples)))
            adj_list.append(np.array([pair for chunk in new_adjs for pair in chunk]).T)
        
        return sample_list, adj_list
    
    def transform(self, sample_list, adj_list):
        
        trans_adj_list, target_index_outs = [], []
        
        base_index_dict = {k:v for v,k in enumerate(sample_list[0])}        
        for i, adjs in enumerate(adj_list):
            target_index_outs.append([base_index_dict[k] for k in adjs[0]])
            
            base_index_dict = {k:v for v,k in enumerate(sample_list[i+1])}
            
            neighbor_index_out, neighbor_index_in = [base_index_dict[k] for k in adjs[0]], [base_index_dict[k] for k in adjs[1]]
            trans_adj_list.append([neighbor_index_out, neighbor_index_in])            
            
        return target_index_outs, trans_adj_list
    
    def forward(self, feats, adj, samples):
        
        sample_list, adj_list = self.sample(adj, samples)
        target_index_outs, trans_adj_list = self.transform(sample_list, adj_list)
        
        x = feats[sample_list[-1]]
        
        for i, layer_homo_atts in enumerate(self.homo_atts):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, trans_adj_list[-i-1], len(sample_list[-i-2]), len(sample_list[-i-1]), target_index_outs[-i-1]) for att in layer_homo_atts], dim=1)
        
        return x
    
    
class HeteroAttLayer(nn.Module):
    
    def __init__(self, nchannel, in_dim, att_dim, device, dropout):
        super(HeteroAttLayer, self).__init__()
        
        self.nchannel = nchannel
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.device = device
        
        self.meta_att = nn.Parameter(torch.zeros(size=(nchannel, att_dim)))
        nn.init.xavier_uniform_(self.meta_att.data, gain=1.414)
        
        self.linear_block = nn.Sequential(nn.Linear(in_dim, att_dim), nn.Tanh())

    def forward(self, hs, nnode):
        
        new_hs = torch.cat([self.linear_block(hs[i]).view(1,nnode,-1) for i in range(self.nchannel)], dim=0)
        
        meta_att = []
        for i in range(self.nchannel):
            meta_att.append(torch.sum(torch.mm(new_hs[i], self.meta_att[i].view(-1,1)).squeeze(1)) / nnode)
        meta_att = torch.stack(meta_att, dim=0)
        meta_att = F.softmax(meta_att, dim=0)
        
        aggre_hid = []
        for i in range(nnode):
            aggre_hid.append(torch.mm(meta_att.view(1,-1), new_hs[:,i,:]))
        aggre_hid = torch.stack(aggre_hid, dim=0).view(nnode, self.att_dim)
        
        return aggre_hid
    
    
class HANModel(nn.Module):
    
    def __init__(self, nchannel, nfeat, nhid, nlabel, nlayer, nheads, neigh_por, dropout, alpha, device):
        super(HANModel, self).__init__()

        self.HomoAttModels = [HomoAttModel(nfeat, nhid, dropout, alpha, device, nheads, nlayer, neigh_por) for i in range(nchannel)]
        self.HeteroAttLayer = HeteroAttLayer(nchannel, nhid*nheads[-1], nhid, device, dropout).to(device)        
        
        for i, homo_att in enumerate(self.HomoAttModels):
            self.add_module('homo_att_{}'.format(i), homo_att)
        self.add_module('hetero_att', self.HeteroAttLayer)
        
        self.supervised = False
        if nlabel!=0:
            self.supervised = True
            self.LinearLayer = torch.nn.Linear(nhid, nlabel).to(device)
            self.add_module('linear', self.LinearLayer)
        
    def forward(self, x, adjs, samples):
        
        homo_out = []
        for i, homo_att in enumerate(self.HomoAttModels):
            homo_out.append(homo_att(x, adjs[i], samples))
        homo_out = torch.stack(homo_out, dim=0)
        aggre_hid = self.HeteroAttLayer(homo_out, len(samples))
        
        if self.supervised:
            pred = self.LinearLayer(aggre_hid)
        else:
            pred = None
        
        return aggre_hid, pred
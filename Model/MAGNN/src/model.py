import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import numpy as np


class MAGNN_mptype_layer(nn.Module):
    
    def __init__(self, etypes, odim, device, nhead, dropout, rvec, rtype='RotatE0', alpha=0.01):
        super(MAGNN_mptype_layer, self).__init__()
        
        self.etypes = etypes
        self.odim = odim
        self.nhead = nhead
        self.rvec = rvec
        self.rtype = rtype
        
        # rnn-like metapath instance aggregator
        # consider multiple attention heads
        if rtype == 'gru':
            self.rnn = nn.GRU(odim, nhead * odim).to(device)
        elif rtype == 'lstm':
            self.rnn = nn.LSTM(odim, nhead * odim).to(device)
        elif rtype == 'bi-gru':
            self.rnn = nn.GRU(odim, nhead * odim // 2, bidirectional=True).to(device)
        elif rtype == 'bi-lstm':
            self.rnn = nn.LSTM(odim, nhead * odim // 2, bidirectional=True).to(device)
        elif rtype == 'linear':
            self.rnn = nn.Linear(odim, nhead * odim).to(device)
        elif rtype == 'max-pooling':
            self.rnn = nn.Linear(odim, nhead * odim).to(device)
        elif rtype == 'neighbor-linear':
            self.rnn = nn.Linear(odim, nhead * odim).to(device)
            
        # node-level attention
        # attention considers the center node embedding
        self.attn1 = nn.Linear(odim, nhead, bias=False).to(device)
        self.attn2 = nn.Parameter(torch.empty(size=(1, nhead, odim))).to(device)
        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.attn_drop = nn.Dropout(dropout) if dropout>0 else lambda x: x 
        
    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}        
    
    def forward(self, g, mpinstances, iftargets, input_node_features):

        edata = []
        for mpinstance in mpinstances:
            edata.append(torch.stack([input_node_features[node] for node in mpinstance]))
        edata = torch.stack(edata)
        center_node_feat = torch.clone(edata[:, -1, :])

        # apply rnn to metapath-based feature sequence
        if self.rtype == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rtype == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rtype == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.odim, self.nhead).permute(0, 2, 1).reshape(
                -1, self.nhead * self.odim).unsqueeze(dim=0)
        elif self.rtype == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.odim, self.nhead).permute(0, 2, 1).reshape(
                -1, self.nhead * self.odim).unsqueeze(dim=0)
        elif self.rtype == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'TransE0' or self.rtype == 'TransE1':
            rvec = self.rvec
            if self.rtype == 'TransE0':
                rvec = torch.stack((rvec, -rvec), dim=1)
                rvec = rvec.reshape(self.rvec.shape[0] * 2, self.rvec.shape[1])  # etypes x odim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + rvec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'RotatE0' or self.rtype == 'RotatE1':
            rvec = F.normalize(self.rvec, p=2, dim=2)
            if self.rtype == 'RotatE0':
                rvec = torch.stack((rvec, rvec), dim=1)
                rvec[:, 1, :, 1] = -rvec[:, 1, :, 1]
                rvec = rvec.reshape(self.rvec.shape[0] * 2, self.rvec.shape[1], 2)  # etypes x odim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_rvec = torch.zeros([edata.shape[1], self.odim // 2, 2], device=edata.device)
            final_rvec[-1, :, 0] = 1
            for i in range(final_rvec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_rvec[i, :, 0] = final_rvec[i + 1, :, 0].clone() * rvec[self.etypes[i], :, 0] -\
                                           final_rvec[i + 1, :, 1].clone() * rvec[self.etypes[i], :, 1]
                    final_rvec[i, :, 1] = final_rvec[i + 1, :, 0].clone() * rvec[self.etypes[i], :, 1] +\
                                           final_rvec[i + 1, :, 1].clone() * rvec[self.etypes[i], :, 0]
                else:
                    final_rvec[i, :, 0] = final_rvec[i + 1, :, 0].clone()
                    final_rvec[i, :, 1] = final_rvec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_rvec[i, :, 0] -\
                        edata[:, i, :, 1].clone() * final_rvec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_rvec[i, :, 1] +\
                        edata[:, i, :, 1].clone() * final_rvec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'neighbor':
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)
            
        eft = hidden.permute(1, 0, 2).view(-1, self.nhead, self.odim)

        a1 = self.attn1(center_node_feat)
        a2 = (eft * self.attn2).sum(dim=-1)
        a = (a1 + a2).unsqueeze(dim=-1)
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})

        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))

        targets = np.where(iftargets[:,1]==1)[0]
        target_features = g.ndata['ft'][targets]

        return iftargets[targets,0], target_features
        
        
class MAGNN_ntype_layer(nn.Module):
    
    def __init__(self, mptype_etypes, odim, adim, device, nhead, dropout, rvec, rtype='RotatE0'):
        super(MAGNN_ntype_layer, self).__init__()
        
        self.odim = odim
        self.nhead = nhead
        
        # metapath-specific layers
        self.MAGNN_mptype_layers = {}
        for mptype, etypes in mptype_etypes.items():
            self.MAGNN_mptype_layers[mptype] = MAGNN_mptype_layer(etypes, odim, device, nhead, dropout, rvec, rtype)
            
        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(odim * nhead, adim, bias=True).to(device)
        self.fc2 = nn.Linear(adim, 1, bias=False).to(device)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)   
        
    def forward(self, mptype_g, mptype_mpinstances, mptype_iftargets, input_node_features):

        output_node_features = []
        for mptype in mptype_iftargets:
            targets, output_mptype_node_features = self.MAGNN_mptype_layers[mptype](mptype_g[mptype], mptype_mpinstances[mptype], mptype_iftargets[mptype], input_node_features)
            output_node_features.append(F.elu(output_mptype_node_features).view(-1, self.odim*self.nhead))

        beta = []
        for each in output_node_features:
            fc1 = torch.tanh(self.fc1(each))
            fc2 = self.fc2(torch.mean(fc1, dim=0))
            beta.append(fc2)
        beta = F.softmax(torch.cat(beta, dim=0), dim=0)
        beta = torch.unsqueeze(torch.unsqueeze(beta, dim=-1), dim=-1)
    
        output_node_features = torch.cat([torch.unsqueeze(each, dim=0) for each in output_node_features], dim=0)
        output_node_features = torch.sum(beta * output_node_features, dim=0)

        return targets, output_node_features
        
        
class MAGNN_layer(nn.Module):
    
    def __init__(self, graph_statistics, idim, odim, adim, device, nhead, dropout, rtype='RotatE0'):
        super(MAGNN_layer, self).__init__()
        
        # etype-specific parameters
        rvec = None
        if rtype == 'TransE0':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'] // 2, idim))).to(device)
        elif rtype == 'TransE1':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'], idim))).to(device)
        elif rtype == 'RotatE0':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'] // 2, idim // 2, 2))).to(device)
        elif rtype == 'RotatE1':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'], idim // 2, 2))).to(device)
        if rvec is not None:
            nn.init.xavier_normal_(rvec.data, gain=1.414)
            
        # ntype-specific layer
        self.MAGNN_ntype_layers = {}
        for ntype, mptype_etypes in graph_statistics['ntype_mptype_etypes'].items():
            self.MAGNN_ntype_layers[ntype] = MAGNN_ntype_layer(mptype_etypes, idim, adim, device, nhead, dropout, rvec, rtype)
            
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(idim * nhead, odim, bias=True).to(device)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        
    def forward(self, ntype_mptype_g, ntype_mptype_mpinstances, ntype_mptype_iftargets, input_node_features):

        # ntype-specific layer
        ntype_targets, ntype_output_node_features = [], []
        for ntype in ntype_mptype_g:
            output_targets, output_node_features = self.MAGNN_ntype_layers[ntype](ntype_mptype_g[ntype], ntype_mptype_mpinstances[ntype], ntype_mptype_iftargets[ntype], input_node_features)
            ntype_targets.append(output_targets)
            ntype_output_node_features.append(output_node_features)

        targets = np.concatenate(ntype_targets)
        transformed = F.elu(self.fc(torch.cat(ntype_output_node_features)))
        node_features = {node:feature for node, feature in zip(targets, transformed)}

        return node_features
        
        
class MAGNN(nn.Module):
    
    # graph_statistics: num_etype (hetero edges only), {ntype: idim}, {ntype: mptype: etypes (homo edges are typed None)}
    def __init__(self, graph_statistics, hdim, adim, dropout, device, nlayer, nhead, nlabel=0, ntype_features={}, rtype='RotatE0'):
        super(MAGNN, self).__init__()
        
        self.device = device
        self.attributed = False if len(ntype_features)==0 else True
        self.supervised = False if nlabel==0 else True
        
        # ntype-specific transformation
        self.ntype_transformation = {}
        for ntype, idim in graph_statistics['ntype_idim'].items():        
            if self.attributed:
                self.ntype_transformation[ntype] = (nn.Embedding.from_pretrained(torch.from_numpy(ntype_features[ntype])).to(self.device), 
                                                    nn.Linear(idim, hdim, bias=True).to(self.device))
                nn.init.xavier_normal_(self.ntype_transformation[ntype][1].weight, gain=1.414)
            else:
                self.ntype_transformation[ntype] = nn.Embedding(idim, hdim).to(self.device)
        self.feat_drop = nn.Dropout(dropout) if dropout>0 else lambda x: x
        
        # MAGNN layers
        self.MAGNN_layers = nn.ModuleList()
        for l in range(nlayer):
            self.MAGNN_layers.append(MAGNN_layer(graph_statistics, hdim, hdim, adim, device, nhead, dropout, rtype))
        
        # prediction layer
        if self.supervised:
            self.final = nn.Linear(hdim, nlabel, bias=True).to(self.device)
            nn.init.xavier_normal_(self.final.weight, gain=1.414)
            
    def forward(self, layer_ntype_mptype_g, layer_ntype_mptype_mpinstances, layer_ntype_mptype_iftargets, batch_ntype_orders):

        # ntype-specific transformation
        node_features = {}
        for ntype, node_orders in batch_ntype_orders.items():
            inputs = torch.from_numpy(np.array(list(node_orders.values())).astype(np.int64)).to(self.device)
            if self.attributed:
                transformed = self.ntype_transformation[ntype][1](self.ntype_transformation[ntype][0](inputs))
            else:
                transformed = self.ntype_transformation[ntype](inputs)
            transformed = self.feat_drop(transformed)
            node_features.update({node:feature for node, feature in zip(node_orders, transformed)})
               
        # MAGNN layers
        for l, layer in enumerate(self.MAGNN_layers):
            node_features = layer(layer_ntype_mptype_g[l], layer_ntype_mptype_mpinstances[l], layer_ntype_mptype_iftargets[l], node_features)

        node_preds = {}
        if self.supervised:
            inputs = torch.stack(list(node_features.values()))
            preds = self.final(inputs)
            node_preds.update({node:pred for node, pred in zip(node_features, preds)})
        
        if self.device=='cuda': torch.cuda.empty_cache()

        return node_features, node_preds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import numpy as np

import pickle
import ast

def read_config(conf_name):
    config = {}
    with open(conf_name) as IN:
        config['edges'] = ast.literal_eval(IN.readline())
        config['nodes'] = ast.literal_eval(IN.readline())
        config['types'] = ast.literal_eval(IN.readline())
        for i,x in enumerate(ast.literal_eval(IN.readline())):
            config['edges'][i].append(int(x))        
    assert len(config['edges']) == len(config['types'])
    return config

def load_emb(temp_dir, data_name, emb_path, emb_size, node_types):
    in_mapping = pickle.load(open(temp_dir + data_name +'_in_mapping.p'))
    type_offset = pickle.load(open(temp_dir + data_name + '_offset.p'))
    with open(emb_path, 'r') as INPUT:
        _data = np.zeros((type_offset['sum'], emb_size))
        INPUT.readline()
        INPUT.readline()
        for line in INPUT:
            node = line.strip().split(' ')
            _type, _id = node[0].split(':')
            _index = in_mapping[_type][_id] + type_offset[_type]
            _data[_index, :] = np.asarray(map(lambda x:float(x), node[1:]))
    return _data

def clip_grad_norm(parameters, max_norm, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
        return 1
    else:
        return 0

def clip_sparse_grad_norm(parameters, max_norm, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        for p in parameters:
            param_norm = p.grad.data._values().norm(norm_type, 1)
            if param_norm.max() > max_norm:
                param_norm.clamp_(min=max_norm).div_(max_norm).unsqueeze_(1)
                #1 how often cut
                #2 cut balanced not 
                p.grad.data._values().div_(param_norm)
                return (param_norm > 1.0).sum()
    return 0

class DiagLinear(nn.Module):
    def __init__(self, input_features):
        super(DiagLinear, self).__init__()
        self.input_features = input_features

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(input_features))

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return input * self.weight

class SymmLinear(nn.Module):
    def __init__(self, input_features):
        super(SymmLinear, self).__init__()
        self.input_features = input_features

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(input_features, input_features))

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        print(input.size(), self.weight.size())
        return input * (self.weight.transpose(0, 1) + self.weight)

class DeepSemantics(nn.Module):
    """
    Multi-layer edge metrics
    """
    def __init__(self, in_features, out_features, hidden_features, bias=False, norm=False):
        super(DeepSemantics, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features, bias = bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias = bias)
        
        #self.fc1.weight.data.uniform_(-0.5, 0.5)
        #self.fc2.weight.data.uniform_(-0.5, 0.5)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.norm = norm
        if not bias:
            self.fc1_bn.register_parameter('bias', None)
            self.fc2_bn.register_parameter('bias', None)

    def forward(self, x):
        if x.size(0) == 1 or not self.norm:
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        else:
            x = F.relu(self.fc1_bn(self.fc1(x)))
            return self.fc2_bn(self.fc2(x))
        
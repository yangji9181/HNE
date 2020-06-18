import numpy as np
import pandas as pd
from collections import defaultdict

from utils import *


class Graph():
    
    def __init__(self):
        super(Graph, self).__init__()
        
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: type -> name -> node_id
            node_bacward: type -> list of feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(dict)
        self.node_bacward = defaultdict(list)
        self.node_feature = {}

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}
        
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]
    
    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True
    
    def update_features(self):
        for ntype, nfeatures in self.node_bacward.items():
            self.node_feature[ntype] = pd.DataFrame(nfeatures)        
        del self.node_bacward
    
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())

    
def to_torch(graph, edge_list, feature, time):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_type = []
    edge_index = []
    edge_type = []
    node_feature = []    
    node_time = []
    edge_time = []
    
    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])

    for t in types:
        node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
        node_feature += list(feature[t])
        node_time += list(time[t])        
        
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type += [edge_dict[relation_type]]   
                    '''
                        Our time ranges from 1900 - 2020, largest span is 120.
                    '''
                    edge_time  += [node_time[tid] - node_time[sid] + 120]
                    
    node_feature = torch.FloatTensor(node_feature)
    edge_time = torch.LongTensor(edge_time)
    node_type = torch.LongTensor(node_type)
    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)
    
    return node_feature, node_type, edge_time, edge_type, edge_index


def sample_subgraph(graph, time_range, sampled_depth, sampled_number, inp):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    '''
    layer_data = defaultdict(lambda: {}) # target_type -> target_id -> [ser, time]
    budget = defaultdict(lambda: defaultdict(lambda: [0., 0])) # source_type -> source_id -> [sampled_score, time]
    new_layer_adj = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))) # target_type -> source_type -> relation_type -> [target_id, source_id]
    
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    if source_time > np.max(list(time_range.keys())) or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_time

    
    '''
        First adding the sampled nodes then updating budget.
    '''
    seed_nodes = defaultdict(lambda: defaultdict(int))
    for _type in inp:
        for _id, _time in inp[_type]:
            seed_nodes[_type][_id] = len(layer_data[_type])
            layer_data[_type][_id] = [len(layer_data[_type]), _time]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
    
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)   
    '''###
        Prepare feature, time and adjacency matrix for the sampled graph
    '''    
    feature, times = {}, {}
    for _type in layer_data:
        if len(layer_data[_type]) == 0: continue            
        times[_type] = np.array(list(layer_data[_type].values()))[:,1]
        idxs  = np.array(list(layer_data[_type].keys()))
        feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'emb']), dtype=np.float)
        
    node_dict, node_num = {}, 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])
            
    edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in layer_data[target_type]:
                    target_ser = layer_data[target_type][target_key][0]
                    if target_key not in tesr:
                        continue
                    tesrt = tesr[target_key]
                    for source_key in layer_data[source_type]:
                        source_ser = layer_data[source_type][source_key][0]
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        if source_key in tesrt:
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
                            
    return feature, times, edge_list, node_dict, seed_nodes   


def preprocess(node, links, n_hid, attributed): 
       
    graph = Graph()
    
    ntypes, insize = {}, n_hid
    if attributed=='True': nfeatures = {}
    with open(node, 'r') as file:
        for line in file:
            if attributed=='False': nid, ntype = line[:-1].split('\t')
            elif attributed=='True': 
                nid, ntype, nfeature = line[:-1].split('\t')
                nfeatures[int(nid)] = np.array(nfeature.split(',')).astype(np.float32)
            ntypes[nid] = ntype
    if attributed=='True': insize = len(nfeatures[int(nid)])
    elif attributed=='False': nfeatures = np.random.randn(len(ntypes), n_hid)
    
    linked = set()
    with open(links, 'r') as file:
        for line in file:
            sid, tid, rtype = line[:-1].split('\t')
            graph.add_edge({'id':sid, 'type':ntypes[sid], 'emb':nfeatures[int(sid)]}, {'id':tid, 'type':ntypes[tid], 'emb':nfeatures[int(tid)]}, time=0, relation_type=rtype, directed=True)
            linked.add(sid)
            linked.add(tid)
            
    for nid in ntypes:
        if nid not in linked:
            graph.add_node({'id':nid, 'type':ntypes[nid], 'emb':nfeatures[int(nid)]})
    
    graph.update_features()
    
    return graph, insize

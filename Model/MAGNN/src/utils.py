import dgl
import time
import torch
import random
import numpy as np
import networkx as nx

from collections import defaultdict


def set_seed(seed, device):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device=="cuda":
        torch.cuda.manual_seed(seed)
        
        
def myprint(text):
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + text, flush=True)
    
    
def check_label(node_labels):
    
    random_label = next(iter(node_labels.values()))
    if isinstance(random_label, int):
        return len(np.unique(list(node_labels.values()))), False
    else:
        return len(random_label), True
    
    
# graph_statistics: num_etype (hetero edges only), {ntype: idim}, {ntype: mptype: etypes (homo edges are typed None)}
def read_data(nodefile, linkfile, pathfile, labelfile, attributed, supervised):
    
    graph_statistics = {'num_etype':0, 'ntype_idim':{}, 'ntype_mptype_etypes':defaultdict(dict)}

    node_features, node_type, ntype_set, idim = {}, {}, defaultdict(set), 0
    with open(nodefile, 'r') as file:
        for line in file:
            line = line[:-1].split('\t')
            node, ntype = int(line[0]), int(line[1])
            node_type[node] = ntype
            ntype_set[ntype].add(node)
            
            if attributed=='True': 
                node_features[node] = np.array(line[2].split(',')).astype(np.float32)
                graph_statistics['ntype_idim'][ntype] = len(node_features[node])
        
    node_order, ntype_features = {}, {}
    type_mask = np.zeros(sum([len(each_set) for each_set in ntype_set.values()])).astype(int)
    for ntype, each_set in ntype_set.items():
        each_set = np.sort(list(each_set))
        type_mask[each_set] = ntype
        for nindex, node in enumerate(each_set):
            node_order[node] = nindex
        if attributed=='False':
            graph_statistics['ntype_idim'][ntype] = len(each_set)
        else:
            ntype_features[ntype] = np.vstack([node_features[node] for node in each_set])
    myprint(f'Finish reading nodes')
    
    etype_edges, etype_info, ltype_counter = defaultdict(lambda: defaultdict(set)), {}, 0
    with open(linkfile, 'r') as file:
        for line in file:
            left, right, ltype = list(map(int, line[:-1].split('\t')))
            etype_edges[(node_type[left], node_type[right])][left].add(right)
            etype_edges[(node_type[right], node_type[left])][right].add(left)
            
            for pair in [(left, right), (right, left)]:
                if (node_type[pair[0]], node_type[pair[1]]) not in etype_info:
                    if node_type[pair[0]]!=node_type[pair[1]]:
                        etype_info[(node_type[pair[0]], node_type[pair[1]])] = ltype_counter
                        ltype_counter += 1
                    else:
                        etype_info[(node_type[pair[0]], node_type[pair[1]])] = None
    graph_statistics['num_etype'] = ltype_counter    
    myprint(f'Finish reading links')
                
    mptypes = []
    with open(pathfile, 'r') as file:
        for line in file:
            mptype = line[:-1].replace('\t', '-')
            mptypes.append(mptype)
            ntypes = list(map(int, line[:-1].split('\t')))
            etypes = [etype_info[(ntypes[pointer], ntypes[pointer+1])] for pointer in range(len(ntypes)-1)]
            graph_statistics['ntype_mptype_etypes'][ntypes[-1]][mptype] = etypes
            
    myprint('Graph Statistics')
    for ntype, mptype_etypes in graph_statistics['ntype_mptype_etypes'].items():
        mptype_etypes_info = f'{ntype}: '
        for mptype, etypes in mptype_etypes.items():
            mptype_etypes_info += '{}->[{}]; '.format(mptype, ','.join(map(str, etypes)))
        myprint(mptype_etypes_info)
            
    def recursive_mpinstance(ntypes, level, lefts):
        
        if lefts[-1] not in etype_edges[(ntypes[level], ntypes[level+1])]:
            return None
        
        if level+1==len(ntypes)-1:
            all_mpinstances = []
            for right in etype_edges[(ntypes[level], ntypes[level+1])][lefts[-1]]:
                all_mpinstances.append(lefts + [right])
            return np.array(all_mpinstances).reshape(-1, len(ntypes))
        
        all_mpinstances = []
        for right in etype_edges[(ntypes[level], ntypes[level+1])][lefts[-1]]:
            right_mpinstances = recursive_mpinstance(ntypes, level+1, lefts+[right])
            if right_mpinstances is not None:
                all_mpinstances.append(right_mpinstances)
        if len(all_mpinstances)==0:
            return None                
        return np.vstack(all_mpinstances)
            
    node_mptype_mpinstances = defaultdict(dict)
    for mptype in mptypes:
        ntypes = list(map(int, mptype.split('-')))
        mpinstances = []
        for left in etype_edges[(ntypes[0], ntypes[1])]:
            left_mpinstances = recursive_mpinstance(ntypes, 0, [left])
            if left_mpinstances is not None:
                mpinstances.append(left_mpinstances)
        myprint(f'Finish finding metapath {mptype}')
        if len(mpinstances)>=0:
            mpinstances = np.vstack(mpinstances)
            unique_dsts = defaultdict(list)
            for dindex, dst in enumerate(mpinstances[:,-1]):
                unique_dsts[dst].append(dindex)
            for dst, dindices in unique_dsts.items():
                node_mptype_mpinstances[dst][mptype] = mpinstances[dindices]    
        myprint(f'Finish sorting metapath {mptype}, number of instances: {len(mpinstances)}')           
    myprint('Finish finding metapath instances')
           
    del node_type, ntype_set
    
    node_labels, posi_edges = None, None
    if supervised=='True':
        with open(labelfile, 'r') as file:            
            node_labels, multi = {}, False
            for line in file:
                node, label = line[:-1].split('\t')
                if ',' in label:
                    multi = True
                    node_labels[int(node)] = np.array(label.split(',')).astype(int)
                else:
                    node_labels[int(node)] = int(label)
                    
            if multi:
                label_num = len(np.unique(np.concatenate([node_label for node_label in node_labels.values()])))
                for node in node_labels:
                    multi_label = np.zeros(label_num).astype(np.float32)
                    multi_label[node_labels[node]] = 1
                    node_labels[node] = multi_label
    elif supervised=='False':
        f = lambda x: tuple(sorted(x))
        posi_edges = np.unique([f([left, right]) for left_rights in etype_edges.values() for left, rights in left_rights.items() for right in rights], axis=0)   
    
    return graph_statistics, type_mask, node_labels, node_order, ntype_features, posi_edges, node_mptype_mpinstances


def sample_mpinstances_perntype(targets, node_mptype_mpinstances, sampling):

    mptype_mpinstances = defaultdict(list)    
    for count, target in enumerate(targets):
        if target not in node_mptype_mpinstances: continue
        for mptype, mpinstances in node_mptype_mpinstances[target].items():
            
            if sampling < len(mpinstances):            
                unique, counts = np.unique(mpinstances[:,0], return_counts=True)       
                p = np.repeat(counts**(-1/4), counts)
                p /= np.sum(p)                
                sampled_idx = np.random.choice(len(mpinstances), sampling, replace=False, p=p)
                mptype_mpinstances[mptype].append(mpinstances[mpinstances[:,0].argsort()][sampled_idx])                                    
            else:
                mptype_mpinstances[mptype].append(mpinstances)

    mptype_mpinstances = {mptype:np.vstack(mpinstances) for mptype, mpinstances in mptype_mpinstances.items()}    
    
    return mptype_mpinstances


def prepare_minibatch(targets, node_mptype_mpinstances, type_mask, node_orders, nlayer, sampling, device):
    
    layer_ntype_mptype_g = [defaultdict(dict) for _ in range(nlayer)]
    layer_ntype_mptype_mpinstances = [defaultdict(dict) for _ in range(nlayer)]
    layer_ntype_mptype_iftargets = [defaultdict(dict) for _ in range(nlayer)]    
    for layer_index in range(nlayer):

        ## group target nodes by type
        ntype_targets = defaultdict(set)
        for target in targets:
            ntype_targets[type_mask[target]].add(target)            

        ## sample metapath instances for each ntype
        targets = set()
        for ntype, curr_targets in ntype_targets.items():
            mptype_mpinstances = sample_mpinstances_perntype(curr_targets, node_mptype_mpinstances, sampling)            

            for mptype, mpinstances in mptype_mpinstances.items():

                ng = nx.MultiDiGraph()
                ng.add_nodes_from(curr_targets)
                ng.add_edges_from(np.vstack([mpinstances[:,0], mpinstances[:,-1]]).T)
                g = dgl.from_networkx(ng).to(device)

                iftargets = {src:False for src in mpinstances[:,0]}
                iftargets.update({dst:True for dst in curr_targets})

                layer_ntype_mptype_g[-layer_index-1][ntype][mptype] = g
                layer_ntype_mptype_mpinstances[-layer_index-1][ntype][mptype] = mpinstances
                layer_ntype_mptype_iftargets[-layer_index-1][ntype][mptype] = np.array(sorted(iftargets.items(), key=lambda x: x[0]))

                targets.update(np.unique(mpinstances))


    batch_ntype_orders = defaultdict(dict)
    for target in targets:
        batch_ntype_orders[type_mask[target]][target] = node_orders[target]

    for ntype in batch_ntype_orders:
        batch_ntype_orders[ntype] = {target:order for target, order in sorted(batch_ntype_orders[ntype].items(), key=lambda x: x[1])}
 
    return layer_ntype_mptype_g, layer_ntype_mptype_mpinstances, layer_ntype_mptype_iftargets, batch_ntype_orders


def nega_sampling(num_nodes, posi_edges):

    f = lambda x: tuple(sorted(x))
    
    posi_pool, posi_count = set([f(each) for each in posi_edges]), posi_edges.shape[0]
    nega_edges, nega_count = set(), 0
    while nega_count < posi_count:
        nega_left, nega_right = np.random.choice(np.arange(num_nodes), size=posi_count-int(nega_count/2), replace=True), np.random.choice(np.arange(num_nodes), size=posi_count-int(nega_count/2), replace=True)
        for each_left, each_right in zip(nega_left, nega_right):
            if each_left==each_right: continue
            if f([each_left, each_right]) in posi_pool: continue
            if f([each_left, each_right]) in nega_edges: continue
            nega_edges.add(f([each_left, each_right]))
            nega_count += 1
            if nega_count >= posi_count: break
                
    nega_edges = np.array(list(nega_edges)).astype(np.int32)
    return nega_edges  
    

class Batcher:
    def __init__(self, supervised, batchsize, pool, shuffle=True):
                
        self.supervised = supervised
        self.batchsize = batchsize
        self.shuffle = shuffle
        
        if self.supervised:
            self.datas = np.array(list(pool.keys()))
            self.labels = np.array(list(pool.values()))
        else:
            self.datas = np.hstack(pool).reshape(-1,2,2)
            self.labels = np.vstack([np.ones(len(self.datas)), np.zeros(len(self.datas))]).T.astype(np.float32)
        self.indices = np.arange(len(self.datas))
        self.num_iterations = int(np.ceil(len(self.indices) / self.batchsize))
        
        self.reset()
        
    def next(self):
        
        if self.num_iterations<=self.iter_counter: self.reset()
                
        sampled = self.indices[self.iter_counter*self.batchsize : (self.iter_counter+1)*self.batchsize]
        self.iter_counter += 1
        
        return self.datas[sampled], self.labels[sampled]
    
    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)        
        self.iter_counter = 0

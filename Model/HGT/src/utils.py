import torch
import numpy as np
from collections import defaultdict


def posi_nega(edge_list, node_dict):
    
    node_pool, posi_pool = set(), set()
    for ttype in edge_list:
        for stype in edge_list[ttype]:
            for rtype in edge_list[ttype][stype]:
                if rtype=='self': continue
                for tser, sser in edge_list[ttype][stype][rtype]:
                    tser, sser = tser+node_dict[ttype][0], sser+node_dict[stype][0]
                    posi_pool.add(tuple(sorted([tser, sser])))
                    node_pool.add(tser)
                    node_pool.add(sser)
    
    posi_count = len(posi_pool)
    nega_pool, nega_count = set(), 0
    while nega_count<=posi_count:
        potential_left, potential_right = np.random.choice(list(node_pool), int(len(node_pool)/2), replace=False), np.random.choice(list(node_pool), int(len(node_pool)/2), replace=False)
        for left, right in zip(potential_left, potential_right):
            if left==right: continue
            potential_edge = tuple(sorted([left, right]))
            if potential_edge not in posi_pool and potential_edge not in nega_pool:
                nega_pool.add(potential_edge)
                nega_count += 1
                if nega_count>posi_count: break  
            
    return np.array(list(posi_pool)), np.array(list(nega_pool))


def realign(graph, seed_nodes, node_dict):
    
    reindex = {}
    for _type in seed_nodes:
        for _id in seed_nodes[_type]:
            reindex[graph.node_feature[_type].loc[_id,'id']] = seed_nodes[_type][_id] + node_dict[_type][0]
            
    return reindex


def prepare_output_batch(graph, args):   
    
    ntypes = graph.get_types()
        
    all_nodes = {ntype: np.arange(len(graph.node_feature[ntype])) for ntype in ntypes}
    for each in all_nodes.values():
        np.random.shuffle(each)
    all_nodes = {ntype: np.array_split(each, len(each)//args.batch_size+1) for ntype, each in all_nodes.items()}
    
    sampled = defaultdict(dict)
    for batch_id in range(max([len(all_nodes[ntype]) for ntype in ntypes])):
        for ntype in ntypes:
            sampled[batch_id][ntype] = all_nodes[ntype][batch_id % len(all_nodes[ntype])]
    
    return sampled
        
                        
def output(args, embs):
    
    with open(args.output, 'w') as file:
        file.write(f'size={args.n_hid}, nhead={args.n_heads}, nlayer={args.n_layers}, dropout={args.dropout}, sample_depth={args.sample_depth}, sample_width={args.sample_width}, nepoch={args.n_epoch}, nbatch={args.n_batch}, batch_size={args.batch_size}, attributed={args.attributed}, supervised={args.supervised}\n')
        
        for ori_idx, node_rep in embs.items():
            file.write(f'{ori_idx}\t')
            file.write(' '.join(node_rep.astype(str)))
            file.write('\n')

def load_label(train_file, graph):
    
    labeled_type, nlabel, multi = None, None, False
    train_pool, ori_train_pool, train_label = set(), set(), {}
    with open(train_file, 'r') as file:
        for index, line in enumerate(file):
            if index==0:
                labeled_type, nlabel = line[:-1].split('\t')
                nlabel = int(nlabel)
            else:
                node, label = line[:-1].split('\t')
                ori_train_pool.add(node)
                nid = graph.node_forward[labeled_type][node]
                train_pool.add(nid)
                if multi or ',' in label:
                    multi = True
                    label_array = np.zeros(nlabel).astype(int)
                    label_array[np.array(label.split(',')).astype(int)] = 1
                    train_label[node] = label_array
                else:
                    train_label[node] = int(label)
    
    return train_pool, ori_train_pool, train_label, nlabel, labeled_type, multi
import dgl
import torch
import numpy as np
from collections import defaultdict


def load_supervised(args, link, node, train_pool):
    
    num_nodes, num_rels, train_data = 0, 0, []
    train_indices = defaultdict(list)
    with open(link, 'r') as file:
        for index, line in enumerate(file):
            if index==0:
                num_nodes, num_rels = line[:-1].split(' ')
                num_nodes, num_rels = int(num_nodes), int(num_rels)
                print(f'#nodes: {num_nodes}, #relations: {num_rels}')
            else:
                line = np.array(line[:-1].split(' ')).astype(int)
                train_data.append(line)                
                if line[0] in train_pool:
                    train_indices[line[0]].append(index-1)                    
                if line[-1] in train_pool:
                    train_indices[line[-1]].append(index-1)
    
    if args.attributed=='True':
        node_attri = {}
        with open(node, 'r') as file:
            for line in file:
                line = line[:-1].split('\t')
                node_attri[int(line[0])] = np.array(line[1].split(',')).astype(np.float32)
        return np.array(train_data), num_nodes, num_rels, train_indices, len(train_indices), np.array([node_attri[k] for k in range(len(node_attri))]).astype(np.float32)
    elif args.attributed=='False':    
        return np.array(train_data), num_nodes, num_rels, train_indices, len(train_indices), None


def load_label(train_label):
    
    train_pool, train_labels, all_labels, multi = set(), {}, set(), False
    with open(train_label, 'r') as file:
        for line in file:
            node, label = line[:-1].split('\t')
            node = int(node)
            train_pool.add(node)
            if multi or ',' in label:
                multi = True
                label = np.array(label.split(',')).astype(int)
                for each in label:
                    all_labels.add(label)
                train_labels[node] = label
            else:
                label = int(label)
                train_labels[node] = label
                all_labels.add(label)
    
    return train_pool, train_labels, len(all_labels), multi


def load_unsupervised(args, link, node):
    
    num_nodes, num_rels, train_data = 0, 0, []
    with open(link, 'r') as file:
        for index, line in enumerate(file):
            if index==0:
                num_nodes, num_rels = line[:-1].split(' ')
                num_nodes, num_rels = int(num_nodes), int(num_rels)
                print(f'#nodes: {num_nodes}, #relations: {num_rels}')
            else:
                line = np.array(line[:-1].split(' ')).astype(int)
                train_data.append(line)
    
    if args.attributed=='True':
        node_attri = {}
        with open(node, 'r') as file:
            for line in file:
                line = line[:-1].split('\t')
                node_attri[int(line[0])] = np.array(line[1].split(',')).astype(np.float32)
        return np.array(train_data), num_nodes, num_rels, np.array([node_attri[k] for k in range(len(node_attri))]).astype(np.float32)
    elif args.attributed=='False':
        return np.array(train_data), num_nodes, num_rels, None


def save(args, embs):
    
    with open(f'{args.output}', 'w') as file:
        file.write(f'size={args.n_hidden}, negative={args.negative_sample}, lr={args.lr}, dropout={args.dropout}, regularization={args.regularization}, grad_norm={args.grad_norm}, num_bases={args.n_bases}, num_layers={args.n_layers}, num_epochs={args.n_epochs}, graph_batch_size={args.graph_batch_size}, graph_split_size={args.graph_split_size}, edge_sampler={args.edge_sampler}, supervised={args.supervised}, attributed={args.attributed}\n')
        for index, emb in enumerate(embs):
            file.write(f'{index}\t')
            file.write(' '.join(emb.astype(str)))
            file.write('\n')
    
    return


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    degrees = np.zeros(num_nodes).astype(int)
    for i,triplet in enumerate(triplets):
        degrees[triplet[0]] += 1
        degrees[triplet[2]] += 1

    return degrees

def sample_edge_uniform(degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def add_labeled_edges(edges, train_indices, ntrain, if_train, label_batch_size, batch_index=0):
    
    if if_train:
        sampled_index = set(np.random.choice(np.arange(ntrain), label_batch_size, replace=False))
    else:
        sampled_index = set(np.arange(batch_index*label_batch_size, min(ntrain, (batch_index+1)*label_batch_size)))
        
    new_edges, sampled_nodes = [], set()
    for index, (labeled_node, node_edges) in enumerate(train_indices.items()):
        if index in sampled_index:
            sampled_nodes.add(labeled_node)
            new_edges.append(np.array(node_edges))
    new_edges = np.concatenate(new_edges)
    new_edges = np.unique(np.concatenate([edges, new_edges]))
    
    return new_edges, sampled_nodes

def correct_order(node_id, sampled_nodes, train_labels, multi, nlabel):
    
    matched_labels, matched_index = [], []
    for index, each in enumerate(node_id):
        if each in sampled_nodes:
            if multi: 
                curr_label = np.zeros(nlabel).astype(int)
                curr_label[train_labels[each]] = 1
                matched_labels.append(curr_label)
            else: 
                matched_labels.append(train_labels[each])
            matched_index.append(index)
    
    return np.array(matched_labels), np.array(matched_index)

def generate_sampled_graph_and_labels_supervised(triplets, sample_size, split_size, num_rels, degrees, negative_rate, sampler, train_indices, train_labels, multi, nlabel, ntrain, if_train=True, label_batch_size=512, batch_index=0):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")
        
    edges, sampled_nodes = add_labeled_edges(edges, train_indices, ntrain, if_train, label_batch_size, batch_index)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    matched_labels, matched_index = correct_order(uniq_v, sampled_nodes, train_labels, multi, nlabel)
    
    # negative sampling
#     samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
#                                         negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
#     print("# sampled nodes: {}, # sampled edges: {}".format(len(uniq_v), len(src)*2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels, (src, rel, dst))
    return g, uniq_v, rel, norm, matched_labels, matched_index


def generate_sampled_graph_and_labels_unsupervised(triplets, sample_size, split_size,
                                      num_rels, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
#     print("# sampled nodes: {}, # sampled edges: {}".format(len(uniq_v), len(src)*2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels, (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    return g, rel, norm

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels
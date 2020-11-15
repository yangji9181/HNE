import time
import math
import torch
import argparse

import utils
from model import *

import numpy as np
import torch.nn.functional as F
np.random.seed(1)


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def main(args):
    
    # load graph data
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'start loading...', flush=True)
    if args.supervised=='True':
        train_pool, train_labels, nlabels, multi = utils.load_label(args.label)
        train_data, num_nodes, num_rels, train_indices, ntrain, node_attri = utils.load_supervised(args, args.link, args.node, train_pool)
    elif args.supervised=='False':
        train_data, num_nodes, num_rels, node_attri = utils.load_unsupervised(args, args.link, args.node)
        nlabels = 0
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'finish loading...', flush=True)
    
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    print('check 1', flush=True)
    # create model
    model = TrainModel(node_attri, num_nodes,
                args.n_hidden,
                num_rels, nlabels,
                num_bases=args.n_bases,
                num_hidden_layers=args.n_layers,
                dropout=args.dropout,
                use_cuda=use_cuda,
                reg_param=args.regularization)
    print('check 2', flush=True)
    if use_cuda:
        model.cuda()
    print('check 3', flush=True)
    # build adj list and calculate degrees for sampling
    degrees = utils.get_adj_and_degrees(num_nodes, train_data)
    print('check 4', flush=True)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # training loop
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "start training...", flush=True)
    for epoch in range(args.n_epochs):
        model.train()

        # perform edge neighborhood sampling to generate training graph and data
        if args.supervised=='True':
            g, node_id, edge_type, node_norm, matched_labels, matched_index = \
            utils.generate_sampled_graph_and_labels_supervised(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, degrees, args.negative_sample, args.edge_sampler, 
                train_indices, train_labels, multi, nlabels, ntrain, if_train=True, label_batch_size=args.label_batch_size)
            if multi: matched_labels = torch.from_numpy(matched_labels).float()
            else: matched_labels = torch.from_numpy(matched_labels).long()
        elif args.supervised=='False':        
            g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels_unsupervised(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, degrees, args.negative_sample,
                args.edge_sampler)
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg, g = node_id.cuda(), deg.cuda(), g.to('cuda')
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            if args.supervised=='True': matched_labels = matched_labels.cuda()
            elif args.supervised=='False': data, labels = data.cuda(), labels.cuda()

        embed, pred = model(g, node_id, edge_type, edge_norm)
        if args.supervised=='True': loss = model.get_supervised_loss(pred, matched_labels, matched_index, multi)
        elif args.supervised=='False': loss = model.get_unsupervised_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        optimizer.zero_grad()  
        
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 
              "Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), flush=True)      

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "training done", flush=True)
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "start output...", flush=True)
    model.eval()
    if args.attributed=='True':
        np.random.shuffle(train_data)
        node_emb, node_over = np.zeros((num_nodes, args.n_hidden)), set()
        batch_total = math.ceil(len(train_data)/args.graph_batch_size)
        for batch_num in range(batch_total):

            # perform edge neighborhood sampling to generate training graph and data
            g, old_node_id, edge_type, node_norm, data, labels = \
                utils.generate_sampled_graph_and_labels_unsupervised(
                    train_data, args.graph_batch_size, args.graph_split_size,
                    num_rels, degrees, args.negative_sample,
                    args.edge_sampler)

            # set node/edge feature
            node_id = torch.from_numpy(old_node_id).view(-1, 1).long()
            edge_type = torch.from_numpy(edge_type)
            edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
            if use_cuda:
                node_id, g = node_id.cuda(), g.to('cuda')
                edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()

            embed, _ = model(g, node_id, edge_type, edge_norm)
            node_emb[old_node_id] = embed.detach().cpu().numpy().astype(np.float32)   
        
            for each in old_node_id:
                node_over.add(each)
        
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 
                  f'finish output batch nubmer {batch_num} -> {batch_total}', flush=True)

        utils.save(args, node_emb)
        
    elif args.attributed=='False':
        utils.save(args, model.rgcn.layers[0].embedding.weight.detach().cpu().numpy())
    
    return
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--link", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--node", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--label", type=str, required=True,
            help="dataset to use")
    parser.add_argument('--output', required=True, type=str, 
            help='Output embedding file')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=50,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight bases for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=2000,
            help="number of minimum training epochs")    
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--label-batch-size", type=int, default=512)
    parser.add_argument("--graph-batch-size", type=int, default=200000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=5,
            help="number of negative samples per positive sample")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
            help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--attributed", type=str, default="False")
    parser.add_argument("--supervised", type=str, default="False")

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
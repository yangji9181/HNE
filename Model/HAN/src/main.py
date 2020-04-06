import time
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import *
from utils import *


def parse_args():
    
    parser = argparse.ArgumentParser(description='HAN')
    
    parser.add_argument('--node', type=str, required=True)
    parser.add_argument('--link', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--meta', type=str, required=True)
    
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--nhead', type=str, default='8')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.4)
    
    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)   
    
    parser.add_argument('--attributed', type=str, default="False")
    parser.add_argument('--supervised', type=str, default="False")
    
    return parser.parse_args()


def output(args, embeddings, id_name):
    
    with open(args.output, 'w') as file:
        file.write(f'size={args.size}, nhead={args.nhead}, dropout={args.dropout}, neigh-por={args.neigh_por}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')
        for nid, name in id_name.items():
            file.write('{}\t{}\n'.format(name, ' '.join(embeddings[nid].astype(str))))


def score(criterion, updates, posi, nega, device):
    
    edges = np.vstack([posi, nega])
    labels = torch.from_numpy(np.concatenate([np.ones(len(posi)), np.zeros(len(nega))]).astype(np.float32)).to(device)
    inner = torch.bmm(updates[edges[:,0]][:,None,:], updates[edges[:,1]][:,:,None]).squeeze()
    loss = criterion(inner, labels)
    
    return loss 
    
    
def main():
    
    args = parse_args()
    
    set_seed(args.seed, args.device)
    if args.supervised=='True':
        adjs, id_name, features = load_data_semisupervised(args, args.node, args.link, args.config, list(map(lambda x: int(x), args.meta.split(','))))
        train_pool, train_label, nlabel, multi = load_label(args.label, id_name)
    elif args.supervised=='False':
        adjs, id_name, target_pool, positive_edges, features = load_data_unsupervised(args, args.node, args.link, args.config, list(map(lambda x: int(x), args.meta.split(','))))
        negative_edges = sample(target_pool, positive_edges)
        nlabel = 0
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'finish loading', flush=True)    
    
    nhead = list(map(lambda x: int(x), args.nhead.split(',')))
    nnode, nchannel, nlayer = len(id_name), len(adjs), len(nhead)
    if args.supervised=='False': posi_size, nega_size = len(positive_edges), len(negative_edges)
    if args.attributed=='True': nfeat = features.shape[1]

    model = HANModel(nchannel, nfeat if args.attributed=='True' else args.size, args.size, nlabel, nlayer, nhead, args.neigh_por, args.dropout, args.alpha, args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    embeddings = torch.from_numpy(features).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnode, args.size).astype(np.float32)).to(args.device)
    if args.supervised=='True': 
        train_label = torch.from_numpy(train_label.astype(np.float32)).to(args.device)
    elif args.supervised=='False':
        criterion = torch.nn.BCEWithLogitsLoss()
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'start training', flush=True)
    model.train()
    losses = []
    for epoch in range(args.epochs):
        
        if args.supervised=='True':
            curr_index = np.sort(np.random.choice(np.arange(len(train_pool)), args.batch_size, replace=False))
            curr_batch = train_pool[curr_index]        
            updates, pred = model(embeddings, adjs, curr_batch)
            if multi:
                loss = F.binary_cross_entropy(torch.sigmoid(pred), train_label[curr_index])
            else:
                loss = F.nll_loss(F.log_softmax(pred, dim=1), train_label[curr_index].long())
        elif args.supervised=='False':
            seed_nodes, indices, posi_batch, nega_batch = convert(positive_edges, negative_edges, posi_size, nega_size, args.batch_size)
            updates, _ = model(embeddings, adjs, seed_nodes)
            loss = score(criterion, updates, posi_batch, nega_batch, args.device)
        loss.backward()
        losses.append(loss.item())        
        
        if (epoch+1)%10==0 or epoch+1==args.epochs:
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish epoch {epoch}, loss {np.mean(losses)}', flush=True)
            optimizer.step()
            optimizer.zero_grad()
            losses = []
        else:
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish epoch {epoch}', flush=True)
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'output embedding', flush=True)
    model.eval()
    outbatch_size = 2*args.batch_size
    rounds = math.ceil(nnode/outbatch_size)
    outputs = np.zeros((nnode, args.size)).astype(np.float32)
    for index, i in enumerate(range(rounds)):
        seed_nodes = np.arange(i*outbatch_size, min((i+1)*outbatch_size, nnode))
        embs, _ = model(embeddings, adjs, seed_nodes)
        outputs[seed_nodes] = embs.detach().cpu().numpy()
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish output batch {index} -> {rounds}', flush=True)
    output(args, outputs, id_name)         
    

if __name__ == '__main__':
    main()
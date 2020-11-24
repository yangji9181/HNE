import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from model import MAGNN


def parse_args():
    
    parser = argparse.ArgumentParser(description='MAGNN')
    
    parser.add_argument('--node', type=str, required=True)
    parser.add_argument('--link', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    parser.add_argument('--seed', type=int, default=820)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--hdim', type=int, default=50)
    parser.add_argument('--adim', type=int, default=100)  
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--rtype', type=str, default='RotatE0')
    parser.add_argument('--dropout', type=float, default=0.5)
    
    parser.add_argument('--nepoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--sampling', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    
    parser.add_argument('--attributed', type=str, default="False")
    parser.add_argument('--supervised', type=str, default="False")
    
    return parser.parse_args()


def batch_loss(supervised, multi, node_features, node_preds, batch_targets, batch_labels, device):

    if supervised=='True' and multi==True:
        batch_preds = torch.stack([node_preds[target] for target in batch_targets])
        batch_labels = torch.from_numpy(batch_labels).to(device)
        loss = F.binary_cross_entropy(torch.sigmoid(batch_preds), batch_labels)   
        
    elif supervised=='True' and multi==False:
        batch_preds = torch.stack([node_preds[target] for target in batch_targets])
        batch_labels = torch.from_numpy(batch_labels).to(device)
        loss = F.nll_loss(F.log_softmax(batch_preds, dim=1), batch_labels)

    else:
        posi_inner = torch.bmm(torch.stack([node_features[each] for each in batch_targets[:,0,0]])[:,None,:], torch.stack([node_features[each] for each in batch_targets[:,0,1]])[:,:,None])
        nega_inner = torch.bmm(torch.stack([node_features[each] for each in batch_targets[:,1,0]])[:,None,:], torch.stack([node_features[each] for each in batch_targets[:,1,1]])[:,:,None])
        loss = -torch.mean(F.logsigmoid(posi_inner) + F.logsigmoid(-nega_inner))
    
    return loss


def main():
    
    args = parse_args()    
    set_seed(args.seed, args.device)
    
    myprint('Start reading data')
    ## if supervised: posi_edges = None; if unsupervised: node_labels = None; if unattributed: ntype_features is empty
    graph_statistics, type_mask, node_labels, node_order, ntype_features, posi_edges, node_mptype_mpinstances = read_data(args.node, args.link, args.path, args.label, args.attributed, args.supervised)
    myprint('Finish reading data')
    
    if args.supervised == 'True':
        nlabel, multi = check_label(node_labels)
        batcher = Batcher(True, args.batchsize, node_labels)
    else:
        nlabel, multi = 0, None
        nega_edges = nega_sampling(len(type_mask), posi_edges)
        batcher = Batcher(False, args.batchsize, [posi_edges, nega_edges])
        
    magnn = MAGNN(graph_statistics, args.hdim, args.adim, args.dropout, args.device, args.nlayer, args.nhead, nlabel, ntype_features, args.rtype)
    optimizer = torch.optim.Adam(magnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    myprint('Start training')
    magnn.train()
    for epoch in range(args.nepoch):            

        batch_targets, batch_labels = batcher.next()
        layer_ntype_mptype_g, layer_ntype_mptype_mpinstances, layer_ntype_mptype_iftargets, batch_ntype_orders = prepare_minibatch(set(batch_targets.flatten()), node_mptype_mpinstances, type_mask, node_order, args.nlayer, args.sampling, args.device)
        
        batch_node_features, batch_node_preds = magnn(layer_ntype_mptype_g, layer_ntype_mptype_mpinstances, layer_ntype_mptype_iftargets, batch_ntype_orders)
        
        loss = batch_loss(args.supervised, multi, batch_node_features, batch_node_preds, batch_targets, batch_labels, args.device)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()                      
            
        myprint(f'Finish Epoch: {epoch}, Loss: {loss.item()}')
            
        del batch_node_features, batch_node_preds, loss
        if args.device=='cuda': torch.cuda.empty_cache()
    
    myprint('Output embedding')
    outfile = open(args.output, 'w')
    outfile.write(f'size={args.hdim}, nhead={args.nhead}, dropout={args.dropout}, sampling={args.sampling}, lr={args.lr}, batch-size={args.batchsize}, epochs={args.nepoch}, attributed={args.attributed}, supervised={args.supervised}\n')
        
    ntype_nodes = defaultdict(list)
    for node, ntype in enumerate(type_mask):
        ntype_nodes[ntype].append(node)
    
    magnn.eval()
    with torch.no_grad():
        for ntype, features in magnn.ntype_transformation.items():
            if args.attributed=='True': features = features[1](features[0].weight)
            else: features = features.weight
            features = features.detach().cpu().numpy()
            for order, feature in enumerate(features):
                outfile.write('{}\t{}\n'.format(ntype_nodes[ntype][order], ' '.join(feature.astype(str))))
            myprint(f'Output ntype {ntype}')
            
            del features
            if args.device=='cuda': torch.cuda.empty_cache()
                          
    outfile.close()
    
    
if __name__=='__main__':
    main()
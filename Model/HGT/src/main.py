import time
import argparse
import numpy as np
np.random.seed(1)

import multiprocessing as mp
from warnings import filterwarnings
filterwarnings("ignore")

from data import *
from utils import *
from model import *


def parse_args():
    
    parser = argparse.ArgumentParser(description='HGT')

    parser.add_argument('--node', type=str)
    parser.add_argument('--link', type=str)
    parser.add_argument('--label', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--cuda', type=int)      
    
    parser.add_argument('--n_hid', type=int, default=50)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--sample_depth', type=int, default=6)
    parser.add_argument('--sample_width', type=int, default=128)
    
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--n_pool', type=int, default=4)    
    parser.add_argument('--n_batch', type=int, default=32) 
    parser.add_argument('--repeat', type=int, default=2) 
    parser.add_argument('--batch_size', type=int, default=256)    
    parser.add_argument('--clip', type=float, default=0.25) 
    
    parser.add_argument('--attributed', type=str, default='False')
    parser.add_argument('--supervised', type=str, default='False')

    return parser.parse_args()


def sample_batch(seed, ver, batch_num=None):
    
    np.random.seed(seed)
    
    seed_nodes, ntypes = {}, graph.get_types()
    for ntype in ntypes:
        type_nnodes = len(graph.node_feature[ntype])
        if ver=='eval':
            batch_sampled = sampled[batch_num][ntype]
        elif ver=='train' and args.supervised=='False':
            batch_sampled = np.random.choice(np.arange(type_nnodes), min(type_nnodes, args.batch_size//len(ntypes)), replace=False)
        elif ver=='train' and args.supervised=='True':
            if ntype==labeled_type:
                batch_sampled = np.random.choice(np.array(list(train_pool)), min(len(train_pool), args.batch_size//len(ntypes)), replace=False)
            else:
                batch_sampled = np.random.choice(np.arange(type_nnodes), min(type_nnodes, args.batch_size//len(ntypes)), replace=False)
        seed_nodes[ntype] = np.vstack([batch_sampled, np.full(len(batch_sampled),0)]).T

    feature, times, edge_list, node_dict, seed_nodes = sample_subgraph(graph, {0:True}, args.sample_depth, args.sample_width, seed_nodes)
    node_feature, node_type, edge_time, edge_type, edge_index = to_torch(graph, edge_list, feature, times)
    posi, nega = posi_nega(edge_list, node_dict)
    reindex = realign(graph, seed_nodes, node_dict)

    return node_feature, node_type, edge_time, edge_type, edge_index, posi, nega, reindex


current_batch = 0
def prepare_data(pool, ver):

    global current_batch
    
    jobs = []
    for batch_id in np.arange(args.n_batch):
        
        if ver=='train':
            p = pool.apply_async(sample_batch, args=(batch_id, ver))
        elif ver=='eval':
            if current_batch>=total_steps: break
            p = pool.apply_async(sample_batch, args=(batch_id, ver, current_batch))
            current_batch += 1            
        jobs.append(p)        
        
    return jobs


def score(criterion, node_rep, posi, nega, device):
    
    edges = np.vstack([posi, nega])
    labels = torch.from_numpy(np.concatenate([np.ones(len(posi)), np.zeros(len(nega))]).astype(np.float32)).to(device)
    inner = torch.bmm(node_rep[edges[:,0]][:,None,:], node_rep[edges[:,1]][:,:,None]).squeeze()
    loss = criterion(inner, labels)
    
    return loss


print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'start reading', flush=True)
args = parse_args()    
graph, in_size = preprocess(args.node, args.link, args.n_hid, args.attributed)
nlabel = 0
if args.supervised=='True': train_pool, ori_train_pool, train_label, nlabel, labeled_type, multi = load_label(args.label, graph)
device = torch.device(f'cuda:{args.cuda}') if args.cuda != -1 else torch.device('cpu')

print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'start modeling', flush=True)
model = GNN(in_size, args.n_hid, len(graph.get_types()), len(graph.get_meta_graph())+1, args.n_heads, args.n_layers, args.dropout, nlabel).to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)
criterion = torch.nn.BCEWithLogitsLoss()

train_step = 1500

print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'start training', flush=True)
pool = mp.Pool(args.n_pool)
jobs = prepare_data(pool, 'train')

for epoch in np.arange(args.n_epoch) + 1:
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'start epoch {epoch}', flush=True)
    
    train_data = [job.get() for job in jobs]
    pool.close()
    pool.join()

    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool, 'train')

    losses = []
    model.train()
    torch.cuda.empty_cache()
    for _ in range(args.repeat):            
        for node_feature, node_type, edge_time, edge_type, edge_index, posi, nega, reindex in train_data:

            if args.supervised=='False':
                node_rep, _ = model.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_type.to(device), edge_index.to(device))
                loss = score(criterion, node_rep, posi, nega, device)
            elif args.supervised=='True':
                _, pred_rep = model.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_type.to(device), edge_index.to(device))
                batch_nodes, batch_labels = [], []
                for ori_idx, batch_idx in reindex.items():
                    if ori_idx in ori_train_pool:
                        batch_nodes.append(pred_rep[batch_idx])
                        batch_labels.append(train_label[ori_idx])
                batch_nodes, batch_labels = torch.stack(batch_nodes), torch.from_numpy(np.array(batch_labels)).to(device)
                if multi:
                    loss = F.binary_cross_entropy(torch.sigmoid(batch_nodes), batch_labels)
                else:
                    loss = F.nll_loss(F.log_softmax(batch_nodes, dim=1), batch_labels)
            losses.append(loss.item())

            optimizer.zero_grad() 
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_step += 1
            scheduler.step(train_step)
            
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish epoch {epoch}, loss {np.mean(losses)}', flush=True)           
    
pool.close()
pool.join()


print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'start outputing', flush=True)

model.eval()
sampled = prepare_output_batch(graph, args)
embs, total_steps, output_step = {}, len(sampled), 0

pool = mp.Pool(args.n_pool)
jobs = prepare_data(pool, 'eval')

while len(jobs)>0:

    eval_data = [job.get() for job in jobs]
    pool.close()
    pool.join()
    
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool, 'eval')

    model.eval()
    torch.cuda.empty_cache()
    for node_feature, node_type, edge_time, edge_type, edge_index, posi, nega, reindex in eval_data:
        node_rep, _ = model.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_type.to(device), edge_index.to(device))
        node_rep = node_rep.detach().cpu().numpy()
        
        for ori_idx, batch_idx in reindex.items():
            embs[ori_idx] = node_rep[batch_idx]
        
        output_step += 1
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish output step {output_step} / {total_steps}', flush=True)

pool.close()
pool.join()

print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish output {len(embs)} == {sum([len(graph.node_feature[_type]) for _type in graph.get_types()])}', flush=True)
output(args, embs)
import os
import sys
import json
import math
import time
import pickle
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.utils.logger import Logger, LogLevel
from spodernet.utils.global_config import Config, Backends
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch, CustomTokenizer, TargetIdx2MultiTarget

from model import ConvE

cudnn.benchmark = True
np.set_printoptions(precision=3)


''' Preprocess knowledge graph using spodernet. '''
def preprocess(args, delete_data=False):
    
    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    print('create dataset streamer', flush=True)
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(args.train_path)
    print('create pipeline', flush=True)
    p = Pipeline(args.data, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    print('execute full vocabs', flush=True)
    p.execute(d)
    print('save full vocabs', flush=True)
    p.save_vocabs()

    # process train sets and save them to hdf5
    p.skip_transformation = False        
    d.set_path(args.train_path)
    p.clear_processors()
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
    p.add_post_processor(StreamToHDF5('train', samples_per_file=1000, keys=input_keys))
    print('execute and save train vocabs', flush=True)
    p.execute(d)


def main(args):
    
    if args.preprocess:
        print('start preprocessing', flush=True)
        preprocess(args, delete_data=True)
        print('finish preprocessing', flush=True)
        
    else:
        input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
        p = Pipeline(args.data, keys=input_keys)
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) + ': start loading vocabs', flush=True)
        p.load_vocabs()
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) + ': finish loading vocabs', flush=True)
        vocab = p.state['vocab']
        num_entities = vocab['e1'].num_token

        train_batcher = StreamBatcher(args.data, 'train', args.batch_size, randomize=True, keys=input_keys, loader_threads=args.loader_threads)
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
        train_batcher.at_batch_prepared_observers.insert(1, TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))

#         eta = ETAHook('train', print_every_x_batches=args.log_interval)
#         train_batcher.subscribe_to_events(eta)
#         train_batcher.subscribe_to_start_of_epoch_event(eta)
#         train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=args.log_interval))

        model.cuda()
        model.init()

        total_param_size = []
        params = [value.numel() for value in model.parameters()]
        print(params, flush=True)
        print(np.sum(params), flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) + f': start training with epochs = {args.epochs}', flush=True)
        for epoch in range(args.epochs):
            model.train()
#             sampled_batches = set(np.random.choice(train_batcher.num_batches, args.num_batches, replace=False))
#             print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) + f': start epoch {epoch} with batches = {len(sampled_batches)} out of {train_batcher.num_batches}', flush=True)
#             processed_count = 0
            for i, str2var in enumerate(train_batcher):                  
#                 if i not in sampled_batches: continue
#                 if processed_count%int(args.num_batches/1000)==0: print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) + f': start epoch {epoch} batch {i} = {processed_count}', flush=True)
#                 processed_count += 1
                opt.zero_grad()
                e1 = str2var['e1']
                rel = str2var['rel']
                e2_multi = str2var['e2_multi1_binary'].float()
                e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))

                pred = model.forward(e1, rel)
                loss = model.loss(pred, e2_multi)
                loss.backward()
                opt.step()

#                 train_batcher.state.loss = loss.cpu()
        
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) + f': finish training epoch {epoch}', flush=True)
        
        model.eval()
        output(args, vocab['e1'], model.emb_e.weight.detach().cpu().numpy())
        

def output(args, vocab, embs):
    
    with open(args.emb_path, 'w') as file:
        file.write('size={}, lr={}, lr_decay={}, epochs={}, batch_size={}\n'.format(args.embedding_dim, args.lr, args.lr_decay, args.epochs, args.batch_size))
        for i, emb in enumerate(embs):
            if i==0 or i==1: continue
            file.write('{}\t{}\n'.format(vocab.get_word(i), ' '.join(emb.astype(str))))
            
    return
        
        
def parse_args():
    
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')    
    
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
#     parser.add_argument('--num-batches', type=int, default=36000, help='number of sampled batches (default: 36000)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')    
    parser.add_argument('--log-interval', type=int, default=1000000, help='how many batches to wait before logging training status')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--preprocess', type=int, default=0, help='preprocess')
    
    parser.add_argument('--data', type=str, help='Targeting dataset')
    parser.add_argument('--emb-path', type=str, help='Path to output file')
    parser.add_argument('--train-path', type=str, help='Path to preprocessed training file')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    Config.backend = 'pytorch'
    Config.cuda = True
    Config.embedding_dim = args.embedding_dim

    torch.manual_seed(args.seed)
    main(args)

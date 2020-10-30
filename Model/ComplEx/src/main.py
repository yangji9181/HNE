import time
import argparse
import numpy as np

from config import Trainer
from model import ComplEx
from loss import SoftplusLoss
from strategy import NegativeSampling
from data import TrainDataLoader


def parse_args():
    
    parser = argparse.ArgumentParser(description='ComplEx')
    
    parser.add_argument('--node', type=str, required=True)
    parser.add_argument('--link', type=str, required=True)
    parser.add_argument('--rela', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    parser.add_argument('--dim', type=int, default=50)    
    parser.add_argument('--regul_rate', type=float, default=1.0)
    
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--neg_ent', type=int, default=25) #
    parser.add_argument('--neg_rel', type=int, default=0) #
    
    parser.add_argument('--train_times', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--opt_method', type=str, default='adagrad', choices=['adagrad','adadelta','adam','sgd'])
    parser.add_argument('--if_gpu', type=int, default=1, choices=[0,1])    

    return parser.parse_args()

    
def main():
    
    args = parse_args()
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "Build training dataloader", flush=True)
    
    # dataloader for training
    train_dataloader = TrainDataLoader(
        tri_file = args.link,
        ent_file = args.node,
        rel_file = args.rela,
        batch_size = args.batch_size,
        bern_flag = True,
        filter_flag = True,
        neg_ent = args.neg_ent,
        neg_rel = args.neg_rel
    )
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "Define ComplEx model", flush=True)

    # define the model
    complEx = ComplEx(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = args.dim//2 # half real, half imaginary
    )

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "Define loss function", flush=True)
    
    # define the loss function
    model = NegativeSampling(
        model = complEx, 
        loss = SoftplusLoss(),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = args.regul_rate
    )
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "Start training", flush=True)

    # train the model
    trainer = Trainer(
        model = model, 
        data_loader = train_dataloader, 
        train_times = args.train_times, 
        alpha = args.alpha, 
        use_gpu = args.if_gpu, 
        opt_method = args.opt_method)
    trainer.run()

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "Save the embeddings", flush=True)
    
    # save the embeddings
    res = complEx.get_parameters("numpy")
    with open(args.output, 'w') as file:
        file.write(f'size={args.dim}, regul_rate={args.regul_rate}, batch_size={args.batch_size}, train_times={args.train_times}, alpha={args.alpha}, opt_method={args.opt_method}\n')
        for nid, (real_half, imag_half) in enumerate(zip(res['ent_re_embeddings.weight'], res['ent_im_embeddings.weight'])):
            file.write('{}\t{}\n'.format(nid, ' '.join(map(str, np.concatenate([real_half, imag_half])))))
    

if __name__=='__main__':
    main()
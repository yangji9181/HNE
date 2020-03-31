import argparse
from transform_model import *


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', requried=True, type=str, help='Targeting dataset.', 
                        choices=['DBLP','Freebase','PubMed','Yelp'])
    parser.add_argument('-model', required=True, type=str, help='Targeting model.', 
                        choices=['metapath2vec-ESim','PTE','HIN2Vec','AspEm','HEER','R-GCN','HAN','TransE','DistMult', 'ConvE'])
    parser.add_argument('-attributed', required=True, type=str, help='Only R-GCN and HAN support attributed training.',
                        choices=['True','False'])
    parser.add_argument('-supervised', required=True, type=str, help='Only R-GCN and HAN support semi-supervised training.', 
                        choices=['True','False'])
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    
    print('Transforming {} to {} input format for {}, {} training!'
          .format(args.dataset, args.model, 
               'attributed' if args.attributed=='True' else 'unattributed', 
               'semi-supervised' if args.supervised=='True' else 'unsupervised'))
    
    if args.model=='metapath2vec-ESim': metapath2vec_esim_convert(args.dataset)
    elif args.model=='PTE': pte_convert(args.dataset)
    elif args.model=='HIN2Vec': hin2vec_convert(args.dataset)
    elif args.model=='AspEm': aspem_convert(args.dataset)
    elif args.model=='HEER': heer_convert(args.dataset)
    elif args.model=='R-GCN': rgcn_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='HAN': han_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='TransE': transe_convert(args.dataset)
    elif args.model=='DistMult': distmult_convert(args.dataset)
    elif args.model=='ConvE': conve_convert(args.dataset)    
        
    print('Data transformation finished!')
    
    return


if __name__=='__main__':
    main()
#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'HGT', and 'MAGNN' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' support attributed='True'

dataset='PubMed' # choose from 'DBLP', 'Yelp', 'Freebase', and 'PubMed'
model='PTE' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'HGT', 'MAGNN', 'TransE', 'ComplEx', 'DistMult', and 'ConvE'
task='both' # choose 'nc' for node classification, 'lp' for link prediction, or 'both' for both tasks
attributed='False' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'

python evaluate.py -dataset ${dataset} -model ${model} -task ${task} -attributed ${attributed} -supervised ${supervised}
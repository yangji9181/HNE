#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'MAGNN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' contain node attributes.

dataset='PubMed' # choose from 'DBLP', 'Yelp', 'Freebase', and 'PubMed'
model='PTE' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'MAGNN', 'HGT', 'TransE', 'DistMult', 'ComplEx', and 'ConvE'
attributed='False' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'

mkdir ../Model/${model}/data
mkdir ../Model/${model}/data/${dataset}

python transform.py -dataset ${dataset} -model ${model} -attributed ${attributed} -supervised ${supervised}
#!/bin/bash

# Note: Only 'R-GCN' and 'HAN' support attributed='True' or supervised='True'

dataset='PubMed' # choose from 'DBLP', 'Yelp', 'Freebase', and 'PubMed'
model='PTE' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'TransE', 'DistMult', and 'ConvE'
attributed='False' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'

python transform.py -dataset ${dataset} -model ${model} -attributed ${attributed} -supervised ${supervised}